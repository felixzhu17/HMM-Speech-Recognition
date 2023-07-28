class DataConfig:
    def __init__(self, folder_path, sr, **kwargs):
        self.folder_path = folder_path
        self.sr = sr
        self.window_ms = 10
        self.overlap_pct = 0.25
        self.mel_banks = 20
        self.n_mfcc = 12
        self.test_size = 0.25

        for key, value in kwargs.items():
            setattr(self, key, value)


class HMMConfig:
    def __init__(self, **kwargs):
        self.n_states = 5
        for key, value in kwargs.items():
            setattr(self, key, value)


class NNConfig:
    def __init__(self, input_dim, output_dim, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoding_dim = 64
        self.n_heads = 4
        self.dropout = 0.2
        self.num_epochs = 50
        self.learning_rate = 0.001
        self.batch_size = 32
        for key, value in kwargs.items():
            setattr(self, key, value)
