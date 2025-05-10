import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, nc, nclass, nh):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1),      # conv0
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),              # pool0

            nn.Conv2d(64, 128, 3, 1, 1),     # conv1
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),              # pool1

            nn.Conv2d(128, 256, 3, 1, 1),    # conv2
            nn.ReLU(True),

            nn.Conv2d(256, 256, 3, 1, 1),    # conv3
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),    # pool2

            nn.Conv2d(256, 512, 3, 1, 1),    # conv4
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 512, 3, 1, 1),    # conv5
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),    # pool3

            nn.Conv2d(512, 512, 2, 1, 0),    # conv6
            nn.ReLU(True)
        )

        self.rnn = nn.Sequential(
            nn.LSTM(512, nh, bidirectional=True, batch_first=False),
            nn.LSTM(nh * 2, nh, bidirectional=True, batch_first=False)
        )

        self.fc = nn.Linear(nh * 2, nclass)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, "height after conv should be 1"
        conv = conv.squeeze(2)  # remove height dim
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        rnn_out, _ = self.rnn(conv)
        output = self.fc(rnn_out)
        return output
