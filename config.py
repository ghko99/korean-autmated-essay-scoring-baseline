

config = {
    "nikl":{
        "model_path" : "./model/nikl_model.pth",
        "prompt_key": "prompt_con",
        "train_dataset_path": "./nikl/train.csv",
        "test_dataset_path": "./nikl/test.csv",
        "essay_key": "essay",
        "is_topic_label": True,  # Set to True if you want to include topic labels in the essays
        "max_length": 200,
        "emb_file_path": './emb/',
        "sentence_sep": '#@문장구분#',
        "evaluator": "1",
        "rubric": ["con1", "con2", "con3", "con4", "con5", "org1", "org2", "exp1", "exp2"],
        "num_range": [1 , 2 , 3 , 4 , 5]
    },
    "aihub_v1":{
        "model_path" : "./model/aihub_v1_model.pth",
        "prompt_key": "essay_main_subject",
        "train_dataset_path": "./aihub_v1/train.csv",
        "test_dataset_path": "./aihub_v1/valid.csv",
        "essay_key": "essay",
        "is_topic_label": False,  # Set to True if you want to include topic labels in the essays
        "max_length": 50,
        "emb_file_path": './emb/',
        "sentence_sep": '#@문장구분#',
        "evaluator": "1",
        "rubric": ["exp1", "exp2", "exp3", "org1", "org2","org3", "org4", "con1", "con2", "con3", "con4"],
        "num_range": [1 , 2 , 3]
    }
}