package api

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"strconv"
	"testing"

	"github.com/Mag1cFall/AIStudio2API/internal/aistudio"
)

type decisionTestService struct {
	lastDecision aistudio.DecisionRequest
}

func (s *decisionTestService) Models(context.Context) ([]aistudio.Model, error) { return nil, nil }
func (s *decisionTestService) CountTokens(context.Context, aistudio.TokenCountRequest) (aistudio.TokenCount, error) {
	return aistudio.TokenCount{}, nil
}
func (s *decisionTestService) Generate(context.Context, aistudio.GenerateRequest) (<-chan aistudio.Event, error) {
	return nil, nil
}

func (s *decisionTestService) CreateDecision(_ context.Context, request aistudio.DecisionRequest) (aistudio.DecisionResult, error) {
	s.lastDecision = request
	var answers []aistudio.DecisionAnswer
	for _, q := range request.Questions {
		switch q.Type {
		case "noul":
			answers = append(answers, aistudio.DecisionAnswer{
				Key:        q.Key,
				Type:       "noul",
				Confidence: 0.98,
				Noul:       0.92,
			})
		case "choice":
			choice := ""
			var probs []aistudio.DecisionProbability
			for idx, opt := range q.Options {
				val := 0.2
				if idx == 0 {
					choice = opt.ID
					val = 0.8
				}
				probs = append(probs, aistudio.DecisionProbability{Key: opt.ID, Value: val})
			}
			answers = append(answers, aistudio.DecisionAnswer{
				Key:           q.Key,
				Type:          "choice",
				Confidence:    0.95,
				Choice:        choice,
				Probabilities: probs,
			})
		case "score":
			var probs, legend []aistudio.DecisionProbability
			for idx, lvl := range q.Levels {
				key := strconv.Itoa(idx)
				val := 0.1
				if idx == len(q.Levels)-1 {
					val = 0.9
				}
				probs = append(probs, aistudio.DecisionProbability{Key: key, Value: val})
				legend = append(legend, aistudio.DecisionProbability{Key: key, Description: lvl})
			}
			answers = append(answers, aistudio.DecisionAnswer{
				Key:           q.Key,
				Type:          "score",
				Confidence:    0.91,
				Score:         1.8,
				Probabilities: probs,
				Legend:        legend,
			})
		}
	}
	return aistudio.DecisionResult{
		Answers:         answers,
		InputTokens:     42,
		InputByModality: []aistudio.DecisionModalityTokens{{Modality: "text", Tokens: 42}},
		TotalTokens:     42,
	}, nil
}

func TestDecisionCompatibilityEndpointsAndFormats(t *testing.T) {
	svc := &decisionTestService{}
	handler := NewHandler(svc, Config{APIKey: "test-key"})

	t.Run("google v1beta decisions", func(t *testing.T) {
		body := `{
			"model": "gemini-decision-1",
			"input": "The item arrived broken.",
			"questions": {
				"damaged": {"type": "noul", "instructions": "Is the item damaged?"}
			}
		}`
		req := httptest.NewRequest(http.MethodPost, "/v1beta/decisions", strings.NewReader(body))
		req.Header.Set("x-goog-api-key", "test-key")
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
		}
		var resp struct {
			Answers map[string]struct {
				Type       string  `json:"type"`
				Noul       float64 `json:"noul"`
				Confidence float64 `json:"confidence"`
			} `json:"answers"`
			Usage struct {
				TotalInputTokens int64 `json:"total_input_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatal(err)
		}
		if resp.Answers["damaged"].Noul != 0.92 || resp.Usage.TotalInputTokens != 42 {
			t.Fatalf("unexpected response: %+v", resp)
		}
	})

	t.Run("gemini action createDecision", func(t *testing.T) {
		body := `{
			"input": "I was charged twice.",
			"questions": {
				"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}
			}
		}`
		req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-decision-1:createDecision", strings.NewReader(body))
		req.Header.Set("x-goog-api-key", "test-key")
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
		}
		if svc.lastDecision.Model != "gemini-decision-1" {
			t.Fatalf("model=%q", svc.lastDecision.Model)
		}
	})

	t.Run("openai decisions array format on /v1/decisions", func(t *testing.T) {
		body := `{
			"model": "gemini-decision-1",
			"input": "Double charge on my invoice.",
			"questions": [
				{"type": "predicate", "name": "is_billing", "instructions": "Is this about billing?\n\nTrue means: Payment or invoice issue.\n\nFalse means: Other topics."},
				{"type": "choice", "name": "dept", "instructions": "Pick department", "choices": [{"value": "billing", "description": "Payments"}, {"value": "tech"}]}
			]
		}`
		req := httptest.NewRequest(http.MethodPost, "/v1/decisions", strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer test-key")
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
		}
		var resp struct {
			Model   string `json:"model"`
			Answers []struct {
				Name        string  `json:"name"`
				Type        string  `json:"type"`
				Probability float64 `json:"probability"`
				Choice      string  `json:"choice"`
				Confidence  float64 `json:"confidence"`
			} `json:"answers"`
			Usage struct {
				InputTokens int64 `json:"input_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatal(err)
		}
		if len(resp.Answers) != 2 || resp.Answers[0].Name != "is_billing" || resp.Answers[0].Type != "predicate" || resp.Usage.InputTokens != 42 {
			t.Fatalf("unexpected openai decision response: %s", rec.Body.String())
		}
	})

	t.Run("systemone and openrouter decisions format", func(t *testing.T) {
		for _, endpoint := range []string{"/v1/systemone", "/v1/v1/systemone", "/decisions", "/api/alpha/decisions"} {
			body := `{
				"model": "gemini-decision-1",
				"state": {"file": "server.go", "port": 2048},
				"questions": {
					"matches": {
						"type": "noul",
						"instructions": "Does it define a server port?",
						"criteria": {"true": "Defines port", "false": "Unrelated"}
					}
				}
			}`
			req := httptest.NewRequest(http.MethodPost, endpoint, bytes.NewBufferString(body))
			req.Header.Set("Authorization", "Bearer test-key")
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, req)
			if rec.Code != http.StatusOK {
				t.Fatalf("endpoint=%s status=%d body=%s", endpoint, rec.Code, rec.Body.String())
			}
			var resp struct {
				Answers map[string]struct {
					Type string  `json:"type"`
					Noul float64 `json:"noul"`
				} `json:"answers"`
				Usage struct {
					InputTokens int64 `json:"input_tokens"`
				} `json:"usage"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
				t.Fatal(err)
			}
			if resp.Answers["matches"].Noul != 0.92 || resp.Usage.InputTokens != 42 {
				t.Fatalf("endpoint=%s unexpected response: %s", endpoint, rec.Body.String())
			}
		}
	})
}
