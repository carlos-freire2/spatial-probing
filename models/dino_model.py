class DINOv3Wrapper:
    def __init__(self, model_name: str = "", token=True):
        # TODO initialize model
        raise NotImplementedError()

    def get_hidden_states(self, x: torch.Tensor) -> list[torch.Tensor]:
        # TODO return list of hidden state embeddings
        raise NotImplementedError()
