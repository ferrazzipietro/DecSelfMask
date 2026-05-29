import numpy as np
from matplotlib import pyplot as plt
import torch





from IPython.display import HTML, display
import numpy as np
import matplotlib
from matplotlib.colors import LinearSegmentedColormap, Colormap
import html as ihtml
from math import sqrt
def render_token_importance(tokens, weights, cmap="white_to_green", alpha=0.55, title=None, exponential_factor=5, remove_tokens_list = [], normalize = True):
    """
    Render tokens as inline colored spans according to importance weights.
    - tokens: list[str]
    - weights: list[float] or 1D array, same length as tokens
    - cmap: str or Colormap. Use "white_to_green" for a white→dark-green map.
    - alpha: background alpha for color
    - title: optional title string shown above
    """
    w = np.array(weights, dtype=float)
    w = np.nan_to_num(w, nan=0.0)
    # Normalize to [0,1] (robust to constant vectors)
    if normalize:
        minv, maxv = float(w.min()), float(w.max())
        rng = (maxv - minv) if (maxv - minv) > 1e-12 else 1.0
        w_norm = (w - minv) / rng
    else:
        w_norm = w

    # Resolve colormap (red_to_green uses custom two-color alpha mapping below)
    cmap_obj = None
    if isinstance(cmap, Colormap):
        cmap_obj = cmap
    elif isinstance(cmap, str) and cmap == "white_to_green":
        cmap_obj = LinearSegmentedColormap.from_list(
            "white_to_green", ["#ffffff", "#006400"]
        )
    elif not (isinstance(cmap, str) and cmap == "red_to_green"):
        try:
            cmap_obj = matplotlib.colormaps.get_cmap(cmap)  # modern API
        except Exception:
            cmap_obj = matplotlib.cm.get_cmap(cmap)  # fallback

    # Heuristic: no leading space before some punctuation
    no_space_before = {",", ".", ";", ":", "!", "?", ")", "]", "}", "%", "'s", "'re", "'ve", "'ll", "'d", "'m"}

    minv_raw, maxv_raw = float(w.min()), float(w.max())
    neg_denom = abs(minv_raw) if minv_raw < -1e-12 else 1.0
    pos_denom = maxv_raw if maxv_raw > 1e-12 else 1.0

    spans = []
    prev = None
    for tok, s, s_raw in zip(tokens, w_norm, w):
        for rem_tok in remove_tokens_list:
            if rem_tok in tok:
                tok = tok.replace(rem_tok, "")
        if isinstance(cmap, str) and cmap == "red_to_green":
            if s_raw <= 0:
                # Negative side: always red, alpha decreases toward zero
                token_alpha = min(max(abs(float(s_raw)) / neg_denom, 0.0), 1.0)
                r, g, b = 255, 0, 0
            else:
                # Positive side: always green, alpha increases up to max (alpha=1)
                token_alpha = min(max(float(s_raw) / pos_denom, 0.0), 1.0)
                r, g, b = 0, 100, 0
            bg = f"rgba({r},{g},{b},{token_alpha})"
        else:
            r, g, b, _ = [int(round(255*(x**exponential_factor))) for x in cmap_obj(float(s))]
            bg = f"rgba({r},{g},{b},{alpha})"
        t = ihtml.escape(str(tok))
        # spacing
        space = "" if (t in no_space_before or (prev is None)) else " "
        span = (
            f"{space}<span style=\"background:{bg}; padding:0.06em 0.18em;"
            f" border-radius:0.2em; margin:0.02em 0.06em; display:inline;\">{t}</span>"
        )
        spans.append(span)
        prev = t

    content = "".join(spans)
    title_html = f"<div style='font-weight:600;margin-bottom:6px'>{ihtml.escape(title)}</div>" if title else ""
    html = (
        "<div style='font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,Helvetica Neue,Arial;"
        " line-height:1.8; font-size:15px; max-width:1100px;'>"
        f"{title_html}"
        f"<div style='word-wrap:break-word'>{content}</div>"
        "</div>"
    )
    display(HTML(html))





def plot_att_one_layer(attentions, tokens, layer, token_pos_start, avg_among_heads = True, head = -1):
    if avg_among_heads and head != -1:
        raise ValueError("If avg_among_heads is True, head must be -1")
    if avg_among_heads:
        att = attentions[layer][0][:, token_pos_start:, token_pos_start:].mean(dim=0)  # [T, T]
        print(f"Plotting attention for layer {layer}, token positions from {token_pos_start}, avg_among_heads")
    else:
        att = attentions[layer][0, head][token_pos_start:, token_pos_start:]  # [T, T]
        print(f"Plotting attention for layer {layer}, token positions from {token_pos_start}, head={head}")
    plt.figure(figsize=(8, 8))
    plt.imshow(att.to(torch.float32).cpu().numpy())
    plt.xticks(np.arange(len(tokens[token_pos_start:])), tokens[token_pos_start:], rotation=90)
    plt.yticks(np.arange(len(tokens[token_pos_start:])), tokens[token_pos_start:])
    plt.colorbar(label="Attention weight")
    plt.tight_layout()
    plt.show()
    return tokens, att.to(torch.float32).cpu().numpy()


def plot_att_avg_layers(attentions, tokens, token_pos_start, figsize=(8,8)):
    num_layers = len(attentions)
    if token_pos_start < 0 or token_pos_start >= attentions[0].size(2):
        raise ValueError("token_pos_start is out of bounds")
    att_sum = torch.zeros(
        (attentions[0].size(2) - token_pos_start, attentions[0].size(2) - token_pos_start),
        device=attentions[0].device,
        dtype=attentions[0].dtype,
    )
    for layer in range(num_layers):
        att_sum += attentions[layer][0][:, token_pos_start:, token_pos_start:].mean(dim=0)
    att_avg = att_sum / num_layers
    print(f"Plotting average attention across all layers, token positions from {token_pos_start}")
    plt.figure(figsize=figsize)
    plt.imshow(att_avg.to(torch.float32).cpu().numpy())
    plt.xticks(np.arange(len(tokens[token_pos_start:])), tokens[token_pos_start:], rotation=90)
    plt.yticks(np.arange(len(tokens[token_pos_start:])), tokens[token_pos_start:])
    plt.colorbar(label="Attention weight")
    plt.tight_layout()
    plt.show()

    return tokens, att_avg.to(torch.float32).cpu().numpy()



def print_one_colORrow_heatmap_avg_among_heads(attentions, tokens, layer, token_position, plot_col=True, plot_row=False):
    if plot_col and plot_row:
        raise ValueError("Only one of plot_col or plot_row can be True")
    if plot_col:
        to_plot_avg_over_heads = (
            torch.mean(attentions[layer][0], dim=0)[:, token_position]
            .to(torch.float32)
            .cpu()
            .numpy()
            .tolist()
        )
    if plot_row:
        to_plot_avg_over_heads = (
            torch.mean(attentions[layer][0], dim=0)[token_position, :]
            .to(torch.float32)
            .cpu()
            .numpy()
            .tolist()
        )
    plt.figure(figsize=(20,3))
    plt.scatter(range(len(tokens)), to_plot_avg_over_heads)
    plt.xticks(ticks=range(len(tokens)), labels=tokens, rotation=90)
    plt.xlabel("Token Index")
    plt.ylabel("Attention Weight")
    plt.title(f"How Token {token_position} IMPACTS Attention Weights for Layer {layer} avg over Heads")
    plt.axhline(y=0.0, color='r', linestyle='--')
    if token_position < 0:
        token_position = len(tokens) + token_position
    plt.axvline(x=token_position, color='g', linestyle='--')
    plt.show()
    return tokens, to_plot_avg_over_heads

def print_one_colORrow_heatmap_avg_among_layers(attentions, tokens, token_position, plot_starts_from_token:int, plot_col=True, plot_row=False, which_layers:list=None, fig_size=(24,3)):
    """
    which_layers: list of layer indices to average over. If None, average over all layers.
    """
    num_layers = len(attentions)
    if plot_col and plot_row:
        raise ValueError("Only one of plot_col or plot_row can be True")
    att_sum = torch.zeros(
        (attentions[0].size(2) - plot_starts_from_token,),
        device=attentions[0].device,
        dtype=attentions[0].dtype,
    )
    if not which_layers:
        title_ends = 'avg over Layers'
        which_layers = list(range(num_layers))
    else:
        title_ends = f'layers {which_layers}'
    for layer in which_layers:
        if plot_col:
            att_sum += torch.mean(attentions[layer][0], dim=0)[plot_starts_from_token:,token_position]
        if plot_row:
            att_sum += torch.mean(attentions[layer][0], dim=0)[token_position, plot_starts_from_token:]
    att_avg = att_sum / len(which_layers)
    to_plot_avg_over_layers = (
        att_avg.to(torch.float32).cpu().numpy().tolist()
    )
    plt.figure(figsize=fig_size)
    plt.bar(range(len(tokens[plot_starts_from_token:])), to_plot_avg_over_layers)
    plt.xticks(ticks=range(len(tokens[plot_starts_from_token:])), labels=tokens[plot_starts_from_token:], rotation=90)
    plt.xlabel("Token Index")
    plt.ylabel("Attention Weight")
    if plot_col:
        plt.title(f"How Token {token_position} IMPACTS Attention Weights {title_ends}")
    if plot_row:
        plt.title(f"How Token {token_position} IS IMPACTED Attention Weights {title_ends}")
    plt.axhline(y=0.0, color='r', linestyle='--')
    token_position = token_position - plot_starts_from_token
    if token_position < 0:
        token_position = len(tokens) + token_position
    plt.axvline(x=token_position, color='g', linestyle='--')
    plt.show()
    return tokens[plot_starts_from_token:], to_plot_avg_over_layers

