#!bin/bash
uv run python src/eval/script/run_mcq_eval.py --rag-type subquestion --grade grade_10 --subject dia_ly --max-questions 100 --output-file src/eval_result/baseline/dia_ly.json
uv run python src/eval/script/run_mcq_eval.py --rag-type subquestion --grade grade_10 --subject hoa_hoc --max-questions 100 --output-file src/eval_result/baseline/hoa_hoc.json
uv run python src/eval/script/run_mcq_eval.py --rag-type subquestion --grade grade_10 --subject lich_su --max-questions 100 --output-file src/eval_result/baseline/lich_su.json
uv run python src/eval/script/run_mcq_eval.py --rag-type subquestion --grade grade_10 --subject toan --max-questions 100 --output-file src/eval_result/baseline/toan.json
uv run python src/eval/script/run_mcq_eval.py --rag-type subquestion --grade grade_10 --subject vat_ly --max-questions 100 --output-file src/eval_result/baseline/vat_ly.json
uv run python src/eval/script/run_mcq_eval.py --rag-type subquestion --grade grade_10 --subject ngu_van --max-questions 100 --output-file src/eval_result/baseline/ngu_van.json
uv run python src/eval/script/run_mcq_eval.py --rag-type subquestion --grade grade_10 --subject sinh_hoc --max-questions 100 --output-file src/eval_result/baseline/sinh_hoc.json