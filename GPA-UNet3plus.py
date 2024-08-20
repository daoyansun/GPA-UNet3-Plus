
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ._blocks import Conv1x1, MaxPool2x2, make_norm
from ._utils import KaimingInitMixin



class ConvBlockNested(nn.Layer):
    def __init__(self, in_ch, out_ch, mid_ch):
        super().__init__()
        self.act = nn.PReLU()
        self.conv1 = nn.Conv2D(in_ch, mid_ch, kernel_size=3, padding=1)
        self.bn1 = make_norm(mid_ch)
        self.conv2 = nn.Conv2D(mid_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = make_norm(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        output = self.act(x + identity)
        return output

class ASPPCV(nn.Layer):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.bn = make_norm(out_channels)
        self.act = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ASPPCV2(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, 1)
        self.bn = make_norm(out_channels)
        self.act = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x

class ASPPCVOUT(nn.Layer):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv1x1(in_ch, out_ch)
        self.bn = make_norm(out_ch)
        self.act = nn.PReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act(x)

        return x

class ASPPPooling(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avg = nn.AvgPool2D(1)
        self.conv = nn.Conv2D(in_channels, out_channels, 1)
        self.bn = make_norm(out_channels)
        self.relu = nn.PReLU()

    def forward(self, x):
        x = self.avg(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return paddle.nn.functional.interpolate(x,scale_factor=1,  mode='bilinear')

class SpatialAttention(nn.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = paddle.mean(x, keepdim=True)
        max_out = paddle.max(x, keepdim=True)
        x1 = paddle.concat([avg_out, max_out], 1)
        x1 = self.conv1(x1)
        x1 = self.sigmoid(x1)
        return x + x1

class ChannelAttention(nn.Layer):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AvgPool2D(1)
        self.max_pool = nn.MaxPool2D(1)
        self.fc1 = nn.Conv2D(in_planes, in_planes // ratio, 1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2D(in_planes // ratio, in_planes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class EECAM(nn.Layer):
    def __init__(self, in_channel):
        super(OUT, self).__init__()
        self.conv = Conv1x1(in_channel, in_channel)
        self.bn = make_norm(in_channel)
        self.act = nn.PReLU()
        self.cam = ChannelAttention(in_channel)
        self.conv_out = Conv1x1(filters[0] * 5, 2)

    def forward(self, x):
        x1 = self.cam(x)
        x2 = self.conv(x)
        x2 = self.bn(x2)
        x2 = self.act(x2)
        x_final = x1 + x2
        return self.conv_out(x_final)

class UNet3(nn.Layer, KaimingInitMixin):
    def __init__(self, in_ch=3, out_ch=2, width=32):
        super().__init__()

        filters = (width, width * 2, width * 4, width * 8, width * 16)

        self.conv0_0 = ConvBlockNested(in_ch, filters[0], filters[0])
        self.conv1_0 = ConvBlockNested(filters[0], filters[1], filters[1])
        self.conv2_0 = ConvBlockNested(filters[1], filters[2], filters[2])
        self.conv3_0 = ConvBlockNested(filters[2], filters[3], filters[3])
        self.conv4_0 = ConvBlockNested(filters[3], filters[4], filters[4])
        self.down = MaxPool2x2()
        self.down2 = nn.MaxPool2D(kernel_size=2, stride=(2, 2), padding=(0, 0))
        self.down4 = nn.MaxPool2D(kernel_size=2, stride=(4, 4), padding=(0, 0))
        self.down8 = nn.MaxPool2D(kernel_size=2, stride=(8, 8), padding=(0, 0))
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.conv_cat = ConvBlockNested(filters[0] * 5, filters[0], filters[0])
        self.conv_aspp = ASPPCVOUT(filters[0] * 5, filters[0])
        self.sam = SpatialAttention()
        self.eecam = EECAM(filters[0]*5)


        self.conv_4_0 = ConvBlockNested(filters[4], filters[0], filters[0])
        self.conv_3_0 = ConvBlockNested(filters[3], filters[0], filters[0])
        self.conv_2_0 = ConvBlockNested(filters[2], filters[0], filters[0])
        self.conv_1_0 = ConvBlockNested(filters[1], filters[0], filters[0])
        self.conv_0_0 = ConvBlockNested(filters[0], filters[0], filters[0])

        self.conv0 = ConvBlockNested(filters[0] * 2, filters[0], filters[0])
        self.conv1 = ConvBlockNested(filters[1] * 2, filters[1], filters[1])
        self.conv2 = ConvBlockNested(filters[2] * 2, filters[2], filters[2])
        self.conv3 = ConvBlockNested(filters[3] * 2, filters[3], filters[3])
        self.conv4 = ConvBlockNested(filters[4] * 2, filters[4], filters[4])

        self.asppconv0 = ASPPCV2(filters[0], filters[0])
        self.asppconv1 = ASPPCV(filters[0], filters[0], 6)
        self.asppconv2 = ASPPCV(filters[0], filters[0], 12)
        self.asppconv3 = ASPPCV(filters[0], filters[0], 18)
        self.aspppooling = ASPPPooling(filters[0], filters[0])

    def forward(self, t1, t2):
        x0_0_t1 = self.conv0_0(t1)
        x1_0_t1 = self.conv1_0(self.down(x0_0_t1))
        x2_0_t1 = self.conv2_0(self.down(x1_0_t1))
        x3_0_t1 = self.conv3_0(self.down(x2_0_t1))
        x4_0_t1 = self.conv4_0(self.down(x3_0_t1))

        x0_0_t2 = self.conv0_0(t2)
        x1_0_t2 = self.conv1_0(self.down(x0_0_t2))
        x2_0_t2 = self.conv2_0(self.down(x1_0_t2))
        x3_0_t2 = self.conv3_0(self.down(x2_0_t2))
        x4_0_t2 = self.conv4_0(self.down(x3_0_t2))

        x0_0 = self.conv0(paddle.concat([x0_0_t1, x0_0_t2], 1))
        x1_0 = self.conv1(paddle.concat([x1_0_t1, x1_0_t2], 1))
        x2_0 = self.conv2(paddle.concat([x2_0_t1, x2_0_t2], 1))
        x3_0 = self.conv3(paddle.concat([x3_0_t1, x3_0_t2], 1))
        x4_0 = self.conv4(paddle.concat([x4_0_t1, x4_0_t2], 1))

        # decoder-------------------------------------------------------------

        x0_0_1 = self.conv_0_0(self.sam(self.down8(x0_0)))
        x1_0_1 = self.conv_1_0(self.sam(self.down4(x1_0)))
        x2_0_1 = self.conv_2_0(self.sam(self.down2(x2_0)))
        x3_0_1 = self.conv_3_0(self.sam(x3_0))
        x4_0_1 = self.conv_4_0(self.sam(self.up1(x4_0)))

        x0_0_2 = self.asppconv0(x0_0_1)
        x1_0_2 = self.asppconv1(x1_0_1)
        x2_0_2 = self.asppconv2(x2_0_1)
        x3_0_2 = self.asppconv3(x3_0_1)
        x4_0_2 = self.aspppooling(x4_0_1)

        x3_1 = self.conv_aspp(paddle.concat([x0_0_2, x1_0_2, x2_0_2, x3_0_2, x4_0_2], 1))
        x2_2 = self.conv_cat(paddle.concat(
            [self.conv_0_0(self.up1(x3_1)), self.conv_0_0(self.down4(x0_0)), self.conv_1_0(self.down2(x1_0)),
             self.conv_2_0(x2_0), self.conv_4_0(self.up2(x4_0))], 1))
        x1_3 = self.conv_cat(paddle.concat(
            [self.conv_0_0(self.up2(x3_1)), self.conv_0_0(self.up1(x2_2)), self.conv_0_0(self.down2(x0_0)),
             self.conv_1_0(x1_0), self.conv_4_0(self.up3(x4_0))], 1))
        x0_4 = paddle.concat(
            [self.conv_0_0(self.up1(x1_3)), self.conv_0_0(self.up2(x2_2)), self.conv_0_0(self.up3(x3_1)),
             self.conv_4_0(self.up4(x4_0)), self.conv_0_0(x0_0)], 1)

        x_out = self.eecam(x0_4)

        return x_out


