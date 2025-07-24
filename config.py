

config = {
    "is_topic_label": False,  # Set to True if you want to include topic labels in the essays
    "nikl":{
        "model_path" : "./model/nikl_model.pth",
        "prompt_key": "prompt_con",
        "dataset_path": "./nikl/dataset.csv",
        "essay_key": "essay",
        "max_length": 200,
        "emb_file_path": './emb/',
        "sentence_sep": '#@문장구분#',
        "evaluators": ["evaluator1_score_" , "evaluator2_score_"],
        "rubric": ["con1", "con2", "con3", "con4", "con5", "org1", "org2", "exp1", "exp2"],
        "num_range": 5
    },
    "aihub_v1":{
        "model_path" : "./model/aihub_v1_model.pth",
        "prompt_key": "essay_prompt",
        "dataset_path": "./aihub_v1/dataset.csv",
        "essay_key": "essay",
        "max_length": 400,
        "emb_file_path": './emb/',
        "sentence_sep": '#@문장구분#',
        "evaluators" : [""],
        "rubric": ["grammar", "vocab", "structure", "length", "clarity","novelty", "description"],
        "num_range": 3
    }
}