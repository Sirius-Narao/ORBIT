class Loss():

    def __init__(self, name:str):
        self.name = name
    
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)
    
    def forward(self, y_pred, y_true):
        raise NotImplementedError("Subclasses must implement the forward method")