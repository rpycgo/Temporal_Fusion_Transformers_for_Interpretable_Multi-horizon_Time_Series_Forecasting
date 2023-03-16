import torch.nn as nn
import torch.nn.functional as F

device='cpu'

class TimeDistributed(nn.Module):
    """
    refer: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/3
    """
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


def linear_layer(
    size,
    activation=None,
    use_time_distributed=False,
    use_bias=True,
    device=device,
    **kwargs
    ):
    linear = nn.Linear(
        in_features=kwargs.get('in_features'),
        out_features=kwargs.get('out_features'),
        bias=use_bias,
        device=device
    )

    if use_time_distributed:
        linear = TimeDistributed(linear)

    return linear


def apply_mlp(
        inputs,
        hidden_size,
        output_size,
        output_activation=None,
        hidden_activation='tanh',
        use_time_distributed=False,
    ):
    if use_time_distributed:
        hidden = TimeDistributed(
            nn.Linear(
                in_features=inputs.shape[-1],
                out_features=hidden_size,
            )
        )(inputs)
        hidden = getattr(F, hidden_activation)(hidden)

        hidden = nn.Linear(
            in_features=hidden_size,
            out_features=output_size,
        )(inputs)
        hidden = getattr(F, output_activation)(hidden)

    else:
        hidden = nn.Linear(
            in_features=inputs.shape[-1],
            out_features=hidden_size,
        )(inputs)
        hidden = getattr(F, hidden_activation)(hidden)

        hidden = nn.Linear(
            in_features=hidden_size,
            out_features=output_size,
        )(inputs)
        hidden = getattr(F, output_activation)(hidden)

    return hidden


def apply_gating_layer(
    x,
    hidden_layer_size,
    dropout_rate=None,
    use_time_distributed=True,
    activation=None
    ):
    if dropout_rate:
        x = nn.Dropout(p=dropout_rate)(x)

    if use_time_distributed:
        activation_layer = TimeDistributed(
            nn.Linear(
                in_features=x.shape[-1],
                out_features=hidden_layer_size
            )
        )
        activation_layer = getattr(F, activation)(activation_layer)

        gated_layer = TimeDistributed(
            nn.Linear(
                in_features=x.shape[-1],
                out_features=hidden_layer_size,
            )
        )
        gated_layer = F.sigmoid(gated_layer)
    
    else:
        activation_layer = nn.Linear(
            in_features=x.shape[-1],
            out_features=hidden_layer_size,
        )
        activation_layer = getattr(F, activation)(activation_layer)

        gated_layer = nn.Linear(
            in_features=x.shape[-1],
            out_features=hidden_layer_size,
        )        
        gated_layer = F.sigmoid(gated_layer)
    
    return activation_layer*gated_layer, gated_layer



print('1')