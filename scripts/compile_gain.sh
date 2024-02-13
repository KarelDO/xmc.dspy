# TECH large, IRe (iGain)
# python compile_irera.py --lm_config_path ./lm_config.json --retriever_model_name sentence-transformers/all-mpnet-base-v2 --dataset_name esco_tech_large --infer_signature_name infer_esco --rank_signature_name rank_esco --infer_student_model_name llama-2-13b-chat --infer_teacher_model_name gpt-3.5-turbo-instruct --rank_student_model_name gpt-4-0125-preview --rank_teacher_model_name gpt-4-0125-preview --no_rank --no_rank_compile --infer_compile_metric_name rp50 --rank_compile_metric_name rp10 --prior_A 0 --rank_topk 50 --do_validation --do_test --prior_path ./data/esco/esco_priors.json --ontology_path ./data/esco/skills_en_label.txt --ontology_name esco --optimizer_name end-to-end --infer_compile_name bootstrap-few-shot-gain-iterative --rank_compile_name bootstrap-few-shot --infer_compile_max_bootstrapped_demos 10 --rank_compile_max_bootstrapped_demos 2 --save_predictions

# TECH large, IRe (iGain), Ra (BFS)
python compile_irera.py --lm_config_path ./lm_config.json --retriever_model_name sentence-transformers/all-mpnet-base-v2 --dataset_name esco_tech_large --infer_signature_name infer_esco --rank_signature_name rank_esco --infer_student_model_name llama-2-13b-chat --infer_teacher_model_name gpt-3.5-turbo-instruct --rank_student_model_name gpt-4-0125-preview --rank_teacher_model_name gpt-4-0125-preview --infer_compile_metric_name rp50 --rank_compile_metric_name rp10 --prior_A 0 --rank_topk 50 --do_validation --do_test --prior_path ./data/esco/esco_priors.json --ontology_path ./data/esco/skills_en_label.txt --ontology_name esco --optimizer_name left-to-right --infer_compile_name bootstrap-few-shot-gain-iterative --rank_compile_name bootstrap-few-shot --infer_compile_max_bootstrapped_demos 10 --rank_compile_max_bootstrapped_demos 2 --save_predictions