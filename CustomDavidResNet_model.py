class CustomDavidResNet(nn.Module):
    def __init__(self):
        super(CustomDavidResNet, self).__init__()

        self.prep = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.prep_bn = nn.BatchNorm2d(64)

        self.X1_C1 = nn.Conv2d(
            64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.X1_bn1 = nn.BatchNorm2d(128)

        self.R1_C1 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.R1_bn1 = nn.BatchNorm2d(128)
        self.R1_C2 = nn.Conv2d(128, 128, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.R1_bn2 = nn.BatchNorm2d(128)

        self.L2_C1 = nn.Conv2d(
            128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.L2_bn1 = nn.BatchNorm2d(256)

        self.X2_C1 = nn.Conv2d(
            256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.X2_bn1 = nn.BatchNorm2d(512)

        self.R2_C1 = nn.Conv2d(
            512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.R2_bn1 = nn.BatchNorm2d(512)
        self.R2_C2 = nn.Conv2d(512, 512, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.R2_bn2 = nn.BatchNorm2d(512)
        self.shortcut = nn.Sequential()

        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        # Prep Layer
        X = F.relu(self.prep_bn(self.prep(x)))

        # Layer 1
        X = F.relu(self.X1_bn1(F.max_pool2d(self.X1_C1(X),2)))
        R1_out = F.relu(self.R1_bn2(self.R1_C2(F.relu(self.R1_bn1(self.R1_C1(X))))))
        X = R1_out + self.shortcut(X)

        # Layer 2
        X = F.relu(self.L2_bn1(F.max_pool2d(self.L2_C1(X),2)))

        # Layer 3
        X = F.relu(self.X2_bn1(F.max_pool2d(self.X2_C1(X),2)))
        R2_out = F.relu(self.R2_bn2(self.R2_C2(F.relu(self.R2_bn1(self.R2_C1(X))))))
        X = R2_out + self.shortcut(X)

        # Output Layer
        out = F.max_pool2d(X, 4)
        # out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return F.log_softmax(out, dim=1)
