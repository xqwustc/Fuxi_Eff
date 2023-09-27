class BaseSelector:
    # This module will be wrapped around the outer layer of the CTR training model for feature selection, ultimately
    # returning the importance (in descending order) and feature IDs of each feature.
    def __init__(self ,model,inputs,feature_map):
        # Initialize the selector
        self.model = model  # model for training
        self.inputs = inputs
        self.feature_map = feature_map
        print("BaseSelector initialized")


    def select(self):
        pass
