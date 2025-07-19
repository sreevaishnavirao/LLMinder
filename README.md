LLMinder is your intelligent companion for selecting the best open-source Large Language Model (LLM) tailored to your needs. Instead of overwhelming you with filters and dropdowns, LLMinder uses Google Gemini to understand your task, maps it to the right evaluation benchmarks, and then suggests the top-performing models using live data from the Open LLM Leaderboard.

The tool supports over 100 languages with automatic translation, and only considers trusted, official models that are unflagged and available on Hugging Face. You can also set a size constraint, and LLMinder will return both the overall best model and the best model under your specified parameter limit.

Powered by a Streamlit interface, LLMinder makes it effortless to describe your task and get results instantly. Benchmark data is refreshed every 2 hours using a backend pipeline that syncs the latest leaderboard CSV to Google Cloud Storage.

Gemini's reasoning capability is what sets this apart — rather than matching keywords, it understands context and intent, helping the tool pick from benchmarks like MMLU-Pro, MUSR, GPQA, BBH, and more based on what your task demands.

To get started, clone the repo, install dependencies, add your .env file with GCS and Gemini credentials, and run the app via streamlit run app.py. Everything else — benchmark mapping, multilingual handling, ranking logic — is taken care of under the hood.

Whether you're building a chatbot, solving math problems, or working on domain-specific reasoning, LLMinder finds your perfect LLM match — intelligently, reliably, and fast.