import json
from extract_features import get_test_features

parameter_file = "test_params.json"
params = json.loads(open(parameter_file).read())

x_pred = get_prediction_features(training_file, data_directory, params['vgg_file'], params['gistFile'], params['semF_file'])