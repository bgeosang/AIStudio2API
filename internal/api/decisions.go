package api

import (
	"bytes"
	"cmp"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"slices"
	"strconv"
	"strings"

	"github.com/Mag1cFall/AIStudio2API/internal/aistudio"
)

// decisionAsset 是 System One 请求中的内联图片资源
type decisionAsset struct {
	Type       string `json:"type"`
	Format     string `json:"format"`
	DataBase64 string `json:"data_base64"`
}

// decisionRequest 是 Decisions API 请求：支持 Google Decisions、OpenAI Decisions、OpenRouter Decisions 与 TypeSafe System One 格式
type decisionRequest struct {
	Model     string                   `json:"model"`
	Input     json.RawMessage          `json:"input"`
	State     json.RawMessage          `json:"state"`
	Questions json.RawMessage          `json:"questions"`
	Images    []json.RawMessage        `json:"images"`
	Assets    map[string]decisionAsset `json:"assets"`
}

// decisionQuestion 是一个决策问题，兼容 Google Decisions、OpenAI Decisions 与 System One 字段
type decisionQuestion struct {
	Name         string          `json:"name"`
	ID           string          `json:"id"`
	Key          string          `json:"key"`
	Type         string          `json:"type"`
	Instructions json.RawMessage `json:"instructions"`
	Criteria     json.RawMessage `json:"criteria"`
	Choices      []struct {
		Value       string  `json:"value"`
		Description *string `json:"description"`
	} `json:"choices"`
	Levels []json.RawMessage `json:"levels"`
}

// decisionAnswer 是一个问题的回答，字段顺序同官方 Decisions API
type decisionAnswer struct {
	Type          string        `json:"type"`
	Noul          *float64      `json:"noul,omitempty"`
	Bool          *float64      `json:"bool,omitempty"`
	Choice        *string       `json:"choice,omitempty"`
	Score         *float64      `json:"score,omitempty"`
	Legend        orderedFields `json:"legend,omitempty"`
	Probabilities orderedFields `json:"probabilities,omitempty"`
	Confidence    float64       `json:"confidence"`
}

// decisionUsage 是 Decisions API 的 token 用量，输出始终为 0
type decisionUsage struct {
	InputTokens           *int64           `json:"input_tokens,omitempty"`
	OutputTokens          *int64           `json:"output_tokens,omitempty"`
	TotalInputTokens      int64            `json:"total_input_tokens"`
	InputTokensByModality []map[string]any `json:"input_tokens_by_modality,omitempty"`
	TotalOutputTokens     int64            `json:"total_output_tokens"`
	TotalTokens           int64            `json:"total_tokens"`
}

// decisionMeta 记录请求协议特征以便按对应格式返回响应
type decisionMeta struct {
	arrayMode      bool
	systemOneMode  bool
	questionOrder  []string
	requestedTypes map[string]string
}

// orderedFields 是按切片顺序编码的 JSON 对象成员
type orderedFields []orderedField

// orderedField 是 JSON 对象的一个成员
type orderedField struct {
	key   string
	value any
}

// MarshalJSON 按切片顺序写出对象成员
func (fields orderedFields) MarshalJSON() ([]byte, error) {
	var buffer bytes.Buffer
	buffer.WriteByte('{')
	for index, field := range fields {
		if index > 0 {
			buffer.WriteByte(',')
		}
		key, _ := json.Marshal(field.key)
		value, err := json.Marshal(field.value)
		if err != nil {
			return nil, err
		}
		buffer.Write(key)
		buffer.WriteByte(':')
		buffer.Write(value)
	}
	buffer.WriteByte('}')
	return buffer.Bytes(), nil
}

// handleDecision 以 Gemini Decision 模型回答 POST /v1beta/decisions、/v1/decisions、/v1/systemone 等端点的决策问题
func (s *server) handleDecision(w http.ResponseWriter, r *http.Request) {
	s.handleModelDecision(w, r, "")
}

// handleModelDecision 处理决策请求，defaultModel 为路径中指定的默认模型 ID
func (s *server) handleModelDecision(w http.ResponseWriter, r *http.Request, defaultModel string) {
	service, ok := s.service.(aistudio.DecisionService)
	if !ok {
		writeGeminiError(w, http.StatusBadRequest, "INVALID_ARGUMENT", "decisions are unavailable")
		return
	}
	var request decisionRequest
	if err := decodeJSON(r, &request); err != nil {
		writeGeminiError(w, http.StatusBadRequest, "INVALID_ARGUMENT", err.Error())
		return
	}
	if strings.TrimSpace(request.Model) == "" {
		request.Model = cmp.Or(strings.TrimSpace(defaultModel), "gemini-decision-1")
	}
	decision, meta, err := request.decision()
	if err == nil {
		err = s.inlineDecisionFiles(r, decision.Input)
	}
	if err != nil {
		writeGeminiError(w, http.StatusBadRequest, "INVALID_ARGUMENT", err.Error())
		return
	}
	if r.URL != nil && r.URL.Path != "/v1beta/decisions" && !strings.HasSuffix(r.URL.Path, ":createDecision") {
		meta.systemOneMode = true
	}
	result, err := service.CreateDecision(r.Context(), decision)
	var blocked *aistudio.PromptFeedbackError
	if errors.As(err, &blocked) {
		writeDecisionBlocked(w, blocked)
		return
	}
	if err != nil {
		if shouldWriteRequestError(r, err) {
			writeGeminiRequestError(w, err)
		}
		return
	}
	SetAccessLogGenerationResult(r.Context(), &aistudio.Usage{InputTokens: result.InputTokens, TotalTokens: result.TotalTokens}, 0)
	usage := decisionUsage{TotalInputTokens: result.InputTokens, TotalTokens: result.TotalTokens}
	if meta.arrayMode || meta.systemOneMode {
		zero := int64(0)
		usage.InputTokens = &result.InputTokens
		usage.OutputTokens = &zero
	}
	for _, item := range result.InputByModality {
		usage.InputTokensByModality = append(usage.InputTokensByModality, map[string]any{"modality": item.Modality, "tokens": item.Tokens})
	}
	if meta.arrayMode {
		writeOpenAIDecisionResponse(w, decision.Model, result, meta, usage)
		return
	}
	answers := make(orderedFields, 0, len(result.Answers))
	for _, answer := range result.Answers {
		answers = append(answers, orderedField{key: answer.Key, value: decisionAnswerJSON(answer, meta.requestedTypes[answer.Key])})
	}
	if meta.systemOneMode {
		writeJSON(w, http.StatusOK, struct {
			Model   string        `json:"model"`
			Answers orderedFields `json:"answers"`
			Usage   decisionUsage `json:"usage"`
		}{decision.Model, answers, usage})
		return
	}
	writeJSON(w, http.StatusOK, struct {
		Answers orderedFields `json:"answers"`
		Usage   decisionUsage `json:"usage"`
	}{answers, usage})
}

// writeOpenAIDecisionResponse 写出 OpenAI Decisions 数组格式的 answers 与 usage
func writeOpenAIDecisionResponse(w http.ResponseWriter, model string, result aistudio.DecisionResult, meta decisionMeta, usage decisionUsage) {
	byKey := make(map[string]aistudio.DecisionAnswer, len(result.Answers))
	for _, answer := range result.Answers {
		byKey[answer.Key] = answer
	}
	order := meta.questionOrder
	if len(order) == 0 {
		for _, answer := range result.Answers {
			order = append(order, answer.Key)
		}
	}
	answers := make([]map[string]any, 0, len(order))
	for _, key := range order {
		answer, ok := byKey[key]
		if !ok {
			continue
		}
		switch answer.Type {
		case "noul":
			qType := meta.requestedTypes[key]
			if qType == "" || qType == "noul" {
				qType = "predicate"
			}
			answers = append(answers, map[string]any{
				"name":        key,
				"type":        qType,
				"probability": answer.Noul,
				"noul":        answer.Noul,
				"confidence":  answer.Confidence,
			})
		case "choice":
			probs := slices.Clone(answer.Probabilities)
			slices.SortStableFunc(probs, func(a, b aistudio.DecisionProbability) int { return cmp.Compare(b.Value, a.Value) })
			items := make([]map[string]any, 0, len(probs))
			for _, p := range probs {
				items = append(items, map[string]any{"value": p.Key, "probability": p.Value})
			}
			answers = append(answers, map[string]any{
				"name":          key,
				"type":          "choice",
				"choice":        answer.Choice,
				"probabilities": items,
				"confidence":    answer.Confidence,
			})
		case "score":
			probs := slices.Clone(answer.Probabilities)
			slices.SortFunc(probs, compareDecisionLevel)
			items := make([]map[string]any, 0, len(probs))
			for _, p := range probs {
				var val any = p.Key
				if num, err := strconv.Atoi(p.Key); err == nil {
					val = num
				}
				items = append(items, map[string]any{"value": val, "probability": p.Value})
			}
			answers = append(answers, map[string]any{
				"name":          key,
				"type":          "score",
				"score":         answer.Score,
				"probabilities": items,
				"confidence":    answer.Confidence,
			})
		}
	}
	writeJSON(w, http.StatusOK, struct {
		Model   string           `json:"model"`
		Answers []map[string]any `json:"answers"`
		Usage   decisionUsage    `json:"usage"`
	}{model, answers, usage})
}

// decisionBlockCodes 是官方 Decisions API 输入被拦截时错误 code 直接使用的拦截原因，其余原因为 content_blocked
var decisionBlockCodes = map[string]bool{"safety": true, "prohibited_content": true, "blocklist": true, "image_safety": true}

// writeDecisionBlocked 按官方 Decisions API 以 400 与拦截类别 code 返回输入被拦截的错误
func writeDecisionBlocked(w http.ResponseWriter, blocked *aistudio.PromptFeedbackError) {
	code := blocked.Reason
	if !decisionBlockCodes[code] {
		code = "content_blocked"
	}
	type blockError struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	}
	writeJSON(w, http.StatusBadRequest, struct {
		Error blockError `json:"error"`
	}{blockError{code, blocked.Message}})
}

// decision 把请求转换为上游决策请求，问题按 ID 排序
func (request decisionRequest) decision() (aistudio.DecisionRequest, decisionMeta, error) {
	meta := decisionMeta{requestedTypes: make(map[string]string)}
	inputRaw := request.Input
	if !rawJSONConfigured(inputRaw) && rawJSONConfigured(request.State) {
		inputRaw = request.State
		meta.systemOneMode = true
	}
	input, err := decisionInputWithAssets(inputRaw, request.Images, request.Assets)
	if err != nil {
		return aistudio.DecisionRequest{}, meta, fmt.Errorf("input: %w", err)
	}
	trimmedQuestions := bytes.TrimSpace(request.Questions)
	if len(trimmedQuestions) == 0 {
		return aistudio.DecisionRequest{}, meta, fmt.Errorf("questions must contain at least one question")
	}
	decision := aistudio.DecisionRequest{Model: request.Model, Input: input}
	if trimmedQuestions[0] == '[' {
		meta.arrayMode = true
		var list []decisionQuestion
		if err := json.Unmarshal(trimmedQuestions, &list); err != nil {
			return aistudio.DecisionRequest{}, meta, fmt.Errorf("questions: %w", err)
		}
		if len(list) == 0 {
			return aistudio.DecisionRequest{}, meta, fmt.Errorf("questions must contain at least one question")
		}
		byKey := make(map[string]aistudio.DecisionQuestion, len(list))
		for idx, item := range list {
			key := cmp.Or(strings.TrimSpace(item.Name), strings.TrimSpace(item.ID), strings.TrimSpace(item.Key))
			if key == "" {
				key = fmt.Sprintf("q_%d", idx)
			}
			meta.questionOrder = append(meta.questionOrder, key)
			meta.requestedTypes[key] = strings.ToLower(strings.TrimSpace(item.Type))
			converted, err := item.decision(key)
			if err != nil {
				return aistudio.DecisionRequest{}, meta, fmt.Errorf("questions.%s: %w", key, err)
			}
			byKey[key] = converted
		}
		for _, key := range slices.Sorted(maps.Keys(byKey)) {
			decision.Questions = append(decision.Questions, byKey[key])
		}
		return decision, meta, nil
	}
	var questionMap map[string]decisionQuestion
	if err := json.Unmarshal(trimmedQuestions, &questionMap); err != nil {
		return aistudio.DecisionRequest{}, meta, fmt.Errorf("questions: %w", err)
	}
	if len(questionMap) == 0 {
		return aistudio.DecisionRequest{}, meta, fmt.Errorf("questions must contain at least one question")
	}
	for _, key := range slices.Sorted(maps.Keys(questionMap)) {
		rawQ := questionMap[key]
		meta.questionOrder = append(meta.questionOrder, key)
		meta.requestedTypes[key] = strings.ToLower(strings.TrimSpace(rawQ.Type))
		question, err := rawQ.decision(key)
		if err != nil {
			return aistudio.DecisionRequest{}, meta, fmt.Errorf("questions.%s: %w", key, err)
		}
		decision.Questions = append(decision.Questions, question)
	}
	return decision, meta, nil
}

// decision 转换一个问题的说明与 criteria
func (question decisionQuestion) decision(key string) (aistudio.DecisionQuestion, error) {
	qType := strings.ToLower(strings.TrimSpace(question.Type))
	switch qType {
	case "bool", "boolean", "predicate", "binary":
		qType = "noul"
	}
	result := aistudio.DecisionQuestion{Key: key, Type: qType}
	instructions, err := decisionContents(question.Instructions)
	if err != nil {
		return result, fmt.Errorf("instructions: %w", err)
	}
	for _, content := range instructions {
		if content.Type != "text" || content.Text == nil {
			return result, fmt.Errorf("instructions only accept text content")
		}
		result.Instructions = append(result.Instructions, *content.Text)
	}
	if len(result.Instructions) == 0 {
		return result, fmt.Errorf("instructions are required")
	}
	switch qType {
	case "noul":
		if rawJSONConfigured(question.Criteria) {
			var criteria struct {
				TrueDescription  string `json:"true_description"`
				FalseDescription string `json:"false_description"`
				True             string `json:"true"`
				False            string `json:"false"`
			}
			if err := json.Unmarshal(question.Criteria, &criteria); err != nil {
				return result, fmt.Errorf("criteria: %w", err)
			}
			result.HasCriteria = true
			result.TrueDescription = cmp.Or(criteria.TrueDescription, criteria.True)
			result.FalseDescription = cmp.Or(criteria.FalseDescription, criteria.False)
		} else if len(result.Instructions) == 1 {
			if instr, yesDesc, noDesc, ok := splitOpenAIPredicateInstructions(result.Instructions[0]); ok {
				result.Instructions = []string{instr}
				result.HasCriteria = true
				result.TrueDescription = yesDesc
				result.FalseDescription = noDesc
			}
		}
	case "choice":
		if len(question.Choices) > 0 {
			options := make(map[string]string, len(question.Choices))
			for _, c := range question.Choices {
				desc := ""
				if c.Description != nil {
					desc = *c.Description
				}
				options[c.Value] = desc
			}
			for _, id := range slices.Sorted(maps.Keys(options)) {
				result.Options = append(result.Options, aistudio.DecisionOption{ID: id, Description: options[id]})
			}
		} else {
			var options map[string]*string
			if err := json.Unmarshal(question.Criteria, &options); err != nil {
				return result, fmt.Errorf("criteria: %w", err)
			}
			for _, id := range slices.Sorted(maps.Keys(options)) {
				desc := ""
				if options[id] != nil {
					desc = *options[id]
				}
				result.Options = append(result.Options, aistudio.DecisionOption{ID: id, Description: desc})
			}
		}
	case "score":
		levelsRaw := question.Levels
		if len(levelsRaw) == 0 && rawJSONConfigured(question.Criteria) {
			if err := json.Unmarshal(question.Criteria, &levelsRaw); err != nil {
				return result, fmt.Errorf("criteria: %w", err)
			}
		}
		for _, item := range levelsRaw {
			var text string
			if json.Unmarshal(item, &text) == nil {
				result.Levels = append(result.Levels, text)
				continue
			}
			var labeled struct {
				Label       string `json:"label"`
				Description string `json:"description"`
			}
			if err := json.Unmarshal(item, &labeled); err != nil {
				return result, fmt.Errorf("criteria: %w", err)
			}
			result.Levels = append(result.Levels, cmp.Or(labeled.Label, labeled.Description))
		}
	default:
		return result, fmt.Errorf("type must be noul, choice or score")
	}
	return result, nil
}

// splitOpenAIPredicateInstructions 拆分 OpenAI Decisions predicate 中内嵌的 True means / False means 说明
func splitOpenAIPredicateInstructions(text string) (instr, yesDesc, noDesc string, ok bool) {
	trueIdx := strings.Index(text, "\n\nTrue means: ")
	falseIdx := strings.Index(text, "\n\nFalse means: ")
	if trueIdx < 0 && falseIdx < 0 {
		return "", "", "", false
	}
	cut := len(text)
	if trueIdx >= 0 && trueIdx < cut {
		cut = trueIdx
	}
	if falseIdx >= 0 && falseIdx < cut {
		cut = falseIdx
	}
	instr = strings.TrimSpace(text[:cut])
	if instr == "" {
		return "", "", "", false
	}
	if trueIdx >= 0 {
		start := trueIdx + len("\n\nTrue means: ")
		if falseIdx > trueIdx {
			yesDesc = strings.TrimSpace(text[start:falseIdx])
		} else {
			yesDesc = strings.TrimSpace(text[start:])
		}
	}
	if falseIdx >= 0 {
		start := falseIdx + len("\n\nFalse means: ")
		if trueIdx > falseIdx {
			noDesc = strings.TrimSpace(text[start:trueIdx])
		} else {
			noDesc = strings.TrimSpace(text[start:])
		}
	}
	return instr, yesDesc, noDesc, true
}

// decisionContents 解析字符串、单个内容块或内容块数组
func decisionContents(raw json.RawMessage) ([]interactionInput, error) {
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return []interactionInput{{Type: "text", Text: &text}}, nil
	}
	return interactionList[interactionInput](raw)
}

// decisionInput 把 input 的文本、图片、音频、文档与视频内容块转换为 part
func decisionInput(raw json.RawMessage) ([]aistudio.Part, error) {
	return decisionInputWithAssets(raw, nil, nil)
}

// decisionInputWithAssets 解析多协议决策输入（字符串、Interactions 块、OpenAI/OpenRouter 消息块、System One state 与图片资产）
func decisionInputWithAssets(raw json.RawMessage, topImages []json.RawMessage, assets map[string]decisionAsset) ([]aistudio.Part, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil, fmt.Errorf("at least one content block is required")
	}
	var parts []aistudio.Part
	if contents, err := decisionContents(trimmed); err == nil {
		valid := true
		for _, content := range contents {
			converted, partErr := content.parts()
			if partErr != nil {
				valid = false
				break
			}
			parts = append(parts, converted...)
		}
		if valid && len(parts) > 0 {
			extra, err := parseExtraDecisionImages(topImages, assets)
			if err != nil {
				return nil, err
			}
			return append(parts, extra...), nil
		}
		parts = nil
	}
	if converted, ok, err := parseCompatDecisionInput(trimmed, assets); err != nil {
		return nil, err
	} else if ok && len(converted) > 0 {
		extra, err := parseExtraDecisionImages(topImages, nil)
		if err != nil {
			return nil, err
		}
		return append(converted, extra...), nil
	}
	text := string(trimmed)
	var unquoted string
	if json.Unmarshal(trimmed, &unquoted) == nil {
		text = unquoted
	}
	if strings.TrimSpace(text) != "" {
		parts = append(parts, aistudio.Part{Text: text})
	}
	extra, err := parseExtraDecisionImages(topImages, assets)
	if err != nil {
		return nil, err
	}
	parts = append(parts, extra...)
	if len(parts) == 0 {
		return nil, fmt.Errorf("at least one content block is required")
	}
	return parts, nil
}

// parseCompatDecisionInput 解析 OpenAI Responses / OpenRouter / System One 形状的输入与内嵌图片
func parseCompatDecisionInput(raw []byte, assets map[string]decisionAsset) ([]aistudio.Part, bool, error) {
	if raw[0] == '{' {
		var obj map[string]json.RawMessage
		if json.Unmarshal(raw, &obj) != nil {
			return nil, false, nil
		}
		var imageParts []aistudio.Part
		for key, val := range obj {
			if strings.HasPrefix(key, "__omp_judgment_images") {
				var items []json.RawMessage
				if json.Unmarshal(val, &items) == nil {
					for _, item := range items {
						part, ok, err := parseSingleImageItem(item, assets)
						if err != nil {
							return nil, false, err
						}
						if ok {
							imageParts = append(imageParts, part)
						}
					}
				}
				delete(obj, key)
			}
		}
		cleaned, _ := json.Marshal(obj)
		parts := []aistudio.Part{{Text: string(cleaned)}}
		parts = append(parts, imageParts...)
		for _, assetID := range slices.Sorted(maps.Keys(assets)) {
			if len(imageParts) == 0 {
				if part, ok, err := assetToPart(assets[assetID]); err != nil {
					return nil, false, err
				} else if ok {
					parts = append(parts, part)
				}
			}
		}
		return parts, true, nil
	}
	if raw[0] == '[' {
		var items []json.RawMessage
		if json.Unmarshal(raw, &items) != nil {
			return nil, false, nil
		}
		var parts []aistudio.Part
		var plainItems []json.RawMessage
		for _, item := range items {
			var roleMsg struct {
				Role    string            `json:"role"`
				Content []json.RawMessage `json:"content"`
			}
			if json.Unmarshal(item, &roleMsg) == nil && roleMsg.Role != "" && len(roleMsg.Content) > 0 {
				for _, c := range roleMsg.Content {
					if part, ok, err := parseSingleContentItem(c, assets); err != nil {
						return nil, false, err
					} else if ok {
						parts = append(parts, part)
					}
				}
				continue
			}
			if part, ok, err := parseSingleContentItem(item, assets); err != nil {
				return nil, false, err
			} else if ok {
				parts = append(parts, part)
				continue
			}
			plainItems = append(plainItems, item)
		}
		if len(plainItems) > 0 {
			encoded, _ := json.Marshal(plainItems)
			if len(plainItems) == 1 {
				var single string
				if json.Unmarshal(plainItems[0], &single) == nil {
					parts = append([]aistudio.Part{{Text: single}}, parts...)
					return parts, true, nil
				}
			}
			parts = append([]aistudio.Part{{Text: string(encoded)}}, parts...)
		}
		if len(parts) > 0 {
			return parts, true, nil
		}
	}
	return nil, false, nil
}

func parseSingleContentItem(raw json.RawMessage, assets map[string]decisionAsset) (aistudio.Part, bool, error) {
	if part, ok, err := parseSingleImageItem(raw, assets); err != nil || ok {
		return part, ok, err
	}
	var block struct {
		Type string  `json:"type"`
		Text *string `json:"text"`
	}
	if json.Unmarshal(raw, &block) == nil && (block.Type == "input_text" || block.Type == "text") && block.Text != nil {
		return aistudio.Part{Text: *block.Text}, true, nil
	}
	return aistudio.Part{}, false, nil
}

func parseSingleImageItem(raw json.RawMessage, assets map[string]decisionAsset) (aistudio.Part, bool, error) {
	var str string
	if json.Unmarshal(raw, &str) == nil {
		if asset, ok := assets[str]; ok {
			return assetToPart(asset)
		}
		if strings.HasPrefix(str, "data:") {
			part, err := fileOrInlinePart(str, "")
			if err != nil {
				return aistudio.Part{}, false, err
			}
			return part, true, nil
		}
		return aistudio.Part{}, false, nil
	}
	var block struct {
		Type     string          `json:"type"`
		ImageURL json.RawMessage `json:"image_url"`
		Data     string          `json:"data"`
		MimeType string          `json:"mimeType"`
		MIME     string          `json:"mime_type"`
	}
	if json.Unmarshal(raw, &block) != nil {
		return aistudio.Part{}, false, nil
	}
	if block.Type == "image" && block.Data != "" {
		data, err := base64.StdEncoding.DecodeString(block.Data)
		if err != nil {
			data, err = base64.RawStdEncoding.DecodeString(block.Data)
		}
		if err != nil {
			return aistudio.Part{}, false, err
		}
		return aistudio.Part{InlineData: &aistudio.Blob{MIME: cmp.Or(block.MimeType, block.MIME, "image/png"), Data: data}}, true, nil
	}
	if block.Type == "input_image" || block.Type == "image_url" {
		urlStr := ""
		if json.Unmarshal(block.ImageURL, &urlStr) != nil {
			var wrapped struct {
				URL string `json:"url"`
			}
			_ = json.Unmarshal(block.ImageURL, &wrapped)
			urlStr = wrapped.URL
		}
		if strings.HasPrefix(urlStr, "data:") {
			part, err := fileOrInlinePart(urlStr, "")
			if err != nil {
				return aistudio.Part{}, false, err
			}
			return part, true, nil
		}
		if remoteMediaURL(urlStr) {
			return aistudio.Part{File: &aistudio.FileRef{ID: urlStr, MIME: "image/jpeg"}}, true, nil
		}
	}
	return aistudio.Part{}, false, nil
}

func parseExtraDecisionImages(topImages []json.RawMessage, assets map[string]decisionAsset) ([]aistudio.Part, error) {
	var parts []aistudio.Part
	for _, raw := range topImages {
		part, ok, err := parseSingleImageItem(raw, assets)
		if err != nil {
			return nil, err
		}
		if ok {
			parts = append(parts, part)
		}
	}
	for _, id := range slices.Sorted(maps.Keys(assets)) {
		part, ok, err := assetToPart(assets[id])
		if err != nil {
			return nil, err
		}
		if ok {
			parts = append(parts, part)
		}
	}
	return parts, nil
}

func assetToPart(asset decisionAsset) (aistudio.Part, bool, error) {
	if asset.DataBase64 == "" {
		return aistudio.Part{}, false, nil
	}
	data, err := base64.StdEncoding.DecodeString(asset.DataBase64)
	if err != nil {
		data, err = base64.RawStdEncoding.DecodeString(asset.DataBase64)
	}
	if err != nil {
		return aistudio.Part{}, false, err
	}
	mime := "image/png"
	if strings.EqualFold(asset.Format, "jpeg") || strings.EqualFold(asset.Format, "jpg") {
		mime = "image/jpeg"
	} else if strings.Contains(asset.Format, "/") {
		mime = asset.Format
	}
	return aistudio.Part{InlineData: &aistudio.Blob{MIME: mime, Data: data}}, true, nil
}

// inlineDecisionFiles 把经本服务 Files API 上传的文件下载为内联数据，http 与 https 链接原样交给上游
func (s *server) inlineDecisionFiles(r *http.Request, parts []aistudio.Part) error {
	for index := range parts {
		part := &parts[index]
		if part.File == nil || remoteMediaURL(part.File.ID) {
			continue
		}
		service, ok := s.service.(aistudio.FileService)
		if !ok {
			return fmt.Errorf("file content is unavailable")
		}
		media, err := service.DownloadFile(r.Context(), part.File.ID)
		if err != nil {
			return err
		}
		data, err := io.ReadAll(media.Body)
		closeErr := media.Body.Close()
		if err = cmp.Or(err, closeErr); err != nil {
			return err
		}
		part.InlineData = &aistudio.Blob{MIME: cmp.Or(part.File.MIME, media.MIME), Data: data}
		part.File = nil
	}
	return nil
}

// decisionAnswerJSON 按官方形状写出回答：choice 概率从高到低，score 概率与等级说明按等级序号
func decisionAnswerJSON(answer aistudio.DecisionAnswer, requestedType string) decisionAnswer {
	result := decisionAnswer{Type: answer.Type, Confidence: answer.Confidence}
	probabilities := slices.Clone(answer.Probabilities)
	switch answer.Type {
	case "noul":
		if requestedType == "bool" {
			result.Type = "bool"
			result.Bool = &answer.Noul
		} else {
			result.Noul = &answer.Noul
		}
	case "choice":
		result.Choice = &answer.Choice
		slices.SortStableFunc(probabilities, func(a, b aistudio.DecisionProbability) int { return cmp.Compare(b.Value, a.Value) })
	case "score":
		result.Score = &answer.Score
		legend := slices.Clone(answer.Legend)
		slices.SortFunc(legend, compareDecisionLevel)
		slices.SortFunc(probabilities, compareDecisionLevel)
		for _, item := range legend {
			result.Legend = append(result.Legend, orderedField{key: item.Key, value: item.Description})
		}
	}
	for _, item := range probabilities {
		result.Probabilities = append(result.Probabilities, orderedField{key: item.Key, value: item.Value})
	}
	return result
}

// compareDecisionLevel 按等级序号比较 score 的概率或等级说明
func compareDecisionLevel(a, b aistudio.DecisionProbability) int {
	left, _ := strconv.Atoi(a.Key)
	right, _ := strconv.Atoi(b.Key)
	return cmp.Compare(left, right)
}
