from rag.chain import query

TEST_CASES = [
    {
        "question": "I need a dataset for image classification",
        "expect_keywords": ["image", "classification"],
    },
    {
        "question": "Which datasets are good for medical diagnosis prediction?",
        "expect_keywords": ["health", "medical", "clinical", "disease"],
    },
    {
        "question": "I want a small tabular dataset for regression with no missing values",
        "expect_keywords": ["regression"],
    },
    {
        "question": "Find me a dataset related to natural language processing or text",
        "expect_keywords": ["text", "language", "nlp", "natural"],
    },
    {
        "question": "What datasets are available for time series forecasting?",
        "expect_keywords": ["time", "series", "temporal", "forecasting"],
    },
]

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def run_tests():
    results = []
    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] Q: {case['question']}")
        result = query(case["question"])
        answer = result["answer"].lower()
        sources = result["sources"]

        keyword_hit = any(kw in answer for kw in case["expect_keywords"])
        has_sources = len(sources) > 0

        status = PASS if (keyword_hit and has_sources) else FAIL
        print(f"  Status  : {status}")
        print(f"  Sources : {[s['name'] for s in sources]}")
        print(f"  Answer  : {result['answer'][:200]}...")
        results.append(keyword_hit and has_sources)

    passed = sum(results)
    print(f"\n{'='*50}")
    print(f"Result: {passed}/{len(TEST_CASES)} passed")


if __name__ == "__main__":
    run_tests()
