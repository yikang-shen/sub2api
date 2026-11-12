package service

import (
	"bytes"
	"encoding/json"
	"strings"
	"unsafe"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	blockTypeServerToolUse       = "server_tool_use"
	blockTypeWebSearchToolResult = "web_search_tool_result"
)

// Fast-path byte patterns: both block types only ever appear as quoted JSON
// string values, so a raw substring check is a safe pre-filter regardless of
// key/value spacing.
var (
	patternServerToolUse       = []byte(`"server_tool_use"`)
	patternWebSearchToolResult = []byte(`"web_search_tool_result"`)
)

// FilterWebSearchHistoryBlocks removes web-search content blocks from
// historical messages when the upstream cannot accept them:
//
//  1. Emulation-synthesized blocks — server_tool_use / web_search_tool_result
//     whose tool-use ID carries webSearchToolUseIDPrefix — are fabricated
//     locally by the web-search emulation (gateway_websearch_emulation.go).
//     No upstream ever issued them, so clients replaying the conversation
//     (e.g. Claude Code) poison every follow-up request. They are stripped
//     for all upstreams.
//  2. For passback-required upstreams (DeepSeek/Kimi/GLM …, see
//     ResolveThinkingProtocol) all server_tool_use / web_search_tool_result
//     blocks are stripped: these upstreams only accept
//     text/thinking/image/tool_use/tool_result and reject anything else with
//     400 "invalid value: `server_tool_use`". anthropic-strict and unknown
//     upstreams keep genuine blocks untouched.
//  3. GLM can pair a server_tool_use with an ordinary tool_result in a later
//     record. When its call is stripped, retain that result as labeled text;
//     forwarding the result wrapper alone produces Moonshot's 400
//     "tool_call_id is not found". Unrelated client tool pairs are unchanged.
//
// The emulated assistant turn always carries a trailing text summary, so the
// search context survives the strip. A message whose content would become
// empty gets a placeholder text block (mirroring FilterThinkingBlocksForRetry).
// Returns the original body unchanged when nothing needs stripping.
func FilterWebSearchHistoryBlocks(body []byte, mappedModel string) []byte {
	if !bytes.Contains(body, patternServerToolUse) && !bytes.Contains(body, patternWebSearchToolResult) {
		return body
	}

	stripAll := ResolveThinkingProtocol(mappedModel) == ThinkingProtocolPassbackRequired

	jsonStr := *(*string)(unsafe.Pointer(&body))
	msgsRes := gjson.Get(jsonStr, "messages")
	if !msgsRes.Exists() || !msgsRes.IsArray() {
		return body
	}

	var messages []any
	if err := json.Unmarshal(sliceRawFromBody(body, msgsRes), &messages); err != nil {
		return body
	}

	// Index server calls before rewriting: their results may be in another message.
	strippedServerIDs := make(map[string]bool)
	clientToolIDs := make(map[string]bool)
	for _, msg := range messages {
		msgMap, _ := msg.(map[string]any)
		content, _ := msgMap["content"].([]any)
		for _, block := range content {
			blockMap, _ := block.(map[string]any)
			id, _ := blockMap["id"].(string)
			if id == "" {
				continue
			}
			if blockMap["type"] == "tool_use" {
				clientToolIDs[id] = true
			}
			if blockMap["type"] == blockTypeServerToolUse && shouldStripWebSearchBlock(blockMap, stripAll) {
				strippedServerIDs[id] = true
			}
		}
	}

	modified := false
	for _, msg := range messages {
		msgMap, ok := msg.(map[string]any)
		if !ok {
			continue
		}
		content, ok := msgMap["content"].([]any)
		if !ok {
			continue
		}

		// 延迟分配：只有命中需剥离的块才构建新 slice。
		var newContent []any
		for i, block := range content {
			blockMap, isMap := block.(map[string]any)
			if isMap && blockMap["type"] == "tool_result" {
				id, _ := blockMap["tool_use_id"].(string)
				text, textual := serverToolResultText(blockMap["content"])
				if strippedServerIDs[id] && !clientToolIDs[id] && textual {
					if newContent == nil {
						newContent = make([]any, 0, len(content))
						newContent = append(newContent, content[:i]...)
					}
					label := "[Server tool result]"
					if blockMap["is_error"] == true {
						label = "[Server tool error]"
					}
					newContent = append(newContent, map[string]any{"type": "text", "text": label + "\n" + text})
					continue
				}
			}
			if isMap && shouldStripWebSearchBlock(blockMap, stripAll) {
				if newContent == nil {
					newContent = make([]any, 0, len(content))
					newContent = append(newContent, content[:i]...)
				}
				continue
			}
			if newContent != nil {
				newContent = append(newContent, block)
			}
		}
		if newContent == nil {
			continue
		}
		modified = true
		if len(newContent) == 0 {
			role, _ := msgMap["role"].(string)
			placeholder := "(content removed)"
			if role == "assistant" {
				placeholder = "(assistant content removed)"
			}
			newContent = []any{map[string]any{"type": "text", "text": placeholder}}
		}
		msgMap["content"] = newContent
	}

	if !modified {
		return body
	}

	msgsBytes, err := json.Marshal(messages)
	if err != nil {
		return body
	}
	out, err := sjson.SetRawBytes(body, "messages", msgsBytes)
	if err != nil {
		return body
	}
	return out
}

// serverToolResultText only converts textual results. Media results retain their
// original wrapper so this filter does not silently discard useful image data.
func serverToolResultText(content any) (string, bool) {
	if text, ok := content.(string); ok {
		return text, true
	}
	blocks, ok := content.([]any)
	if !ok {
		return "", false
	}
	texts := make([]string, 0, len(blocks))
	for _, entry := range blocks {
		block, _ := entry.(map[string]any)
		text, ok := block["text"].(string)
		if block["type"] != "text" || !ok {
			return "", false
		}
		texts = append(texts, text)
	}
	return strings.Join(texts, "\n"), true
}

func shouldStripWebSearchBlock(block map[string]any, stripAll bool) bool {
	blockType, _ := block["type"].(string)
	switch blockType {
	case blockTypeServerToolUse:
		if stripAll {
			return true
		}
		id, _ := block["id"].(string)
		return strings.HasPrefix(id, webSearchToolUseIDPrefix)
	case blockTypeWebSearchToolResult:
		if stripAll {
			return true
		}
		id, _ := block["tool_use_id"].(string)
		return strings.HasPrefix(id, webSearchToolUseIDPrefix)
	default:
		return false
	}
}
