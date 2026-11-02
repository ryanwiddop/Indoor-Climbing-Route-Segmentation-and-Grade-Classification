import colorsys
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from joblib import Parallel, delayed
from PIL import Image, ImageOps
from sklearn.metrics import precision_recall_curve, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase_2_xgboost import IOU_MATCH, OUT_DIR, macro_recovery

IMAGE_DIR = "/home/public/rwiddop/images"
FIG_DIR = f"{OUT_DIR}/report"
WALLS = ["02.jpg", "29.jpg", "00.jpg", "23.jpg", "34.jpg", "22.jpg"]
REDUCE = 1
DPI = 200
DIM = 0.70
DESAT = 0.40
FILL_ALPHA = 0.88
MIN_TAG_PT = 4.0
MAX_TAG_PT = 9.0
VOLUME_CATEGORY_ID = 1
HARD_COLOUR_FLOOR = 6.0
HARD_SPATIAL = 6.0

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CRITICAL = "#d03b3b"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
        "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Nimbus Sans"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlepad": 10,
    "legend.frameon": False,
    "figure.autolayout": False,
})

STROKE = [pe.withStroke(linewidth=2.6, foreground="white")]


def tint(colour, amount):
    r, g, b = mcolors.to_rgb(colour)
    if amount >= 0:
        return tuple(c + (1.0 - c) * amount for c in (r, g, b))
    return tuple(c * (1.0 + amount) for c in (r, g, b))


def vivid(colour, boost=1.45):
    r, g, b = mcolors.to_rgb(colour)
    h, l, v = colorsys.rgb_to_hsv(r, g, b)
    return colorsys.hsv_to_rgb(h, min(1.0, l * boost), min(1.0, v * 1.08))


def route_colour(k):
    return vivid(tint(SLOT[k % len(SLOT)], (0.0, -0.34, 0.34)[(k // len(SLOT)) % 3]))


def label(route):
    return str(route)[:-2] if str(route).endswith(".0") else str(route)


def grid_on(ax, axis="y"):
    ax.grid(True, axis=axis, zorder=0)
    ax.set_axisbelow(True)


def spread(points, min_dx, min_dy):
    """Nudge near-coincident label anchors downward so text stays readable."""
    order = sorted(range(len(points)), key=lambda i: (points[i][1], points[i][0]))
    out = list(points)
    for a, i in enumerate(order):
        for j in order[:a]:
            if abs(out[i][0] - out[j][0]) < min_dx and abs(out[i][1] - out[j][1]) < min_dy:
                out[i] = (out[i][0], out[j][1] + min_dy)
    return out


def stack(values, lo, hi, gap):
    """Push labels apart along one axis, preserving order."""
    order = np.argsort(values)
    out = np.array(values, dtype=float)
    edge = lo
    for i in order:
        out[i] = max(out[i], edge)
        edge = out[i] + gap
    if edge - gap > hi:
        edge = hi
        for i in order[::-1]:
            out[i] = min(out[i], edge)
            edge = out[i] - gap
    return out


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = f"{FIG_DIR}/{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


class Data:
    def __init__(self):
        with open(f"{OUT_DIR}/predicted_routes.json") as f:
            res = json.load(f)
        with open(f"{OUT_DIR}/metrics.json") as f:
            self.metrics = json.load(f)
        self.threshold = res["threshold"]
        self.gt = {w: {r: set(v) for r, v in d.items()} for w, d in res["gt"].items()}
        self.pred = {w: {int(k): set(v) for k, v in d.items()} for w, d in res["predicted"].items()}
        self.matches = res["matches"]
        self.pairs = pd.read_csv(f"{OUT_DIR}/all_pairs_with_preds.csv")
        self.holds = pd.read_csv("clab_data.csv").set_index("id")

        with open("annotations.json") as f:
            coco = json.load(f)
        file_of = {i["id"]: i["file_name"] for i in coco["images"]}
        self.polys = defaultdict(dict)
        for a in coco["annotations"]:
            self.polys[file_of[a["image_id"]]][a["id"]] = a["segmentation"]
        self.walls = sorted(self.gt)

    def recovery(self, wall):
        return self.metrics["per_wall_recovery"][wall]

    def gt_order(self, wall):
        return sorted(self.gt[wall], key=float)

    def colour_of_gt(self, wall):
        return {r: route_colour(k) for k, r in enumerate(self.gt_order(wall))}

    def pred_match(self, wall):
        """pred cluster index -> (gt route, iou), only for non-degenerate matches."""
        return {m["pred_ind"]: (m["gt_route"], m["iou"])
                for m in self.matches[wall] if m["iou"] > 0}

    def gt_match(self, wall):
        return {m["gt_route"]: (m["pred_ind"], m["iou"])
                for m in self.matches[wall] if m["iou"] > 0}

    def cohesion(self, wall, groups):
        """Per hold: mean P(same route) against the other members of its own group."""
        pairs = self.pairs[self.pairs["image_file"] == wall]
        ids = sorted(set(pairs["i_id"]) | set(pairs["j_id"]))
        index_of = {h: k for k, h in enumerate(ids)}
        affinity = np.full((len(ids), len(ids)), np.nan)
        rows = pairs["i_id"].map(index_of).to_numpy()
        cols = pairs["j_id"].map(index_of).to_numpy()
        probs = pairs["oof_probs"].to_numpy()
        affinity[rows, cols] = probs
        affinity[cols, rows] = probs

        out = {}
        for holds in groups.values():
            members = [index_of[h] for h in holds if h in index_of]
            for hold in holds:
                if hold not in index_of:
                    continue
                others = [m for m in members if m != index_of[hold]]
                out[hold] = float(np.mean(affinity[index_of[hold], others])) if others else np.nan
        return out

    def cluster_of_hold(self, wall):
        return {h: k for k, holds in self.pred[wall].items() for h in holds}

    def route_of_hold(self, wall):
        return {h: r for r, holds in self.gt[wall].items() for h in holds}


def tag_spec(outline, name, confidence, scale):
    points = np.asarray(outline, dtype=float).reshape(-1, 2) / scale
    text = name if confidence is None or np.isnan(confidence) else f"{name} {confidence:.0%}"
    span = max(float(np.ptp(points[:, 0])), 1.0)
    size = span / (0.62 * len(text)) * 72 / DPI
    if size < MIN_TAG_PT and len(text) > len(name):
        text, size = name, span / (0.62 * len(name)) * 72 / DPI
    x, y = points.mean(axis=0)
    return {"x": float(x), "y": float(y), "text": text,
            "size": float(np.clip(size, MIN_TAG_PT, MAX_TAG_PT))}


def place_tags(ax, tags):
    """Nudge colliding hold tags apart, accounting for each tag's rendered size."""
    px = DPI / 72.0
    for _ in range(2):
        order = sorted(range(len(tags)), key=lambda i: tags[i]["y"])
        for a, i in enumerate(order):
            one = tags[i]
            for j in order[:a]:
                two = tags[j]
                gap_x = 0.31 * px * (one["size"] * len(one["text"]) + two["size"] * len(two["text"]))
                gap_y = 0.68 * px * (one["size"] + two["size"])
                if abs(one["x"] - two["x"]) < gap_x and abs(one["y"] - two["y"]) < gap_y:
                    one["y"] = two["y"] + gap_y
    for tag in tags:
        ax.text(tag["x"], tag["y"], tag["text"], ha="center", va="center", zorder=6,
                fontsize=tag["size"], fontweight="bold", color=INK,
                path_effects=[pe.withStroke(linewidth=tag["size"] * 0.38, foreground="white")])


def draw_masks(ax, polys, groups, colours, hatches, names, conf, scale):
    tags = []
    for key, holds in groups.items():
        colour = colours[key]
        for hold in holds:
            for poly in polys.get(hold, []):
                pts = np.asarray(poly, dtype=float).reshape(-1, 2) / scale
                ax.add_patch(Polygon(pts, closed=True, linewidth=2.4,
                                     facecolor=(*mcolors.to_rgb(colour), FILL_ALPHA),
                                     edgecolor=tint(colour, -0.55),
                                     hatch=hatches.get(key), zorder=3))
            if polys.get(hold):
                tags.append(tag_spec(polys[hold][0], names[key], conf.get(hold), scale))
    place_tags(ax, tags)


def fig_overlay(data, wall):
    image = ImageOps.exif_transpose(Image.open(f"{IMAGE_DIR}/{wall}")).reduce(REDUCE)
    pixels = np.asarray(image.convert("RGB"), dtype=float)
    grey = pixels.mean(axis=2, keepdims=True)
    image = ((pixels * (1 - DESAT) + grey * DESAT) * DIM).astype(np.uint8)

    polys = data.polys[wall]
    colours = data.colour_of_gt(wall)
    pred_match = data.pred_match(wall)
    recovery = data.recovery(wall)
    height, width = image.shape[:2]

    side_by_side = height > width
    panel_w, panel_h = width / DPI, height / DPI
    if side_by_side:
        fig_w, fig_h = 2 * panel_w + 0.25, panel_h + 0.75
        frac_w, frac_h = panel_w / fig_w, panel_h / fig_h
        boxes = [[0, 0.004, frac_w, frac_h],
                 [1 - frac_w, 0.004, frac_w, frac_h]]
    else:
        fig_w, fig_h = panel_w, 2 * panel_h + 1.6
        frac_w, frac_h = 1.0, panel_h / fig_h
        boxes = [[0, 1 - 0.035 - frac_h, frac_w, frac_h],
                 [0, 1 - 0.090 - 2 * frac_h, frac_w, frac_h]]

    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = [fig.add_axes(box) for box in boxes]
    for ax in axes:
        ax.imshow(image, zorder=1)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis("off")

    draw_masks(axes[0], polys, data.gt[wall], colours, {},
               {r: label(r) for r in data.gt[wall]},
               data.cohesion(wall, data.gt[wall]), REDUCE)
    axes[0].set_title(f"ground truth  —  {len(data.gt[wall])} routes", loc="left")

    pred_colours, hatches, names = {}, {}, {}
    for k in data.pred[wall]:
        if k in pred_match:
            route, iou = pred_match[k]
            pred_colours[k] = colours[route]
            names[k] = label(route)
        else:
            pred_colours[k] = mcolors.to_rgb(MUTED)
            hatches[k] = "///"
            names[k] = "?"
    draw_masks(axes[1], polys, data.pred[wall], pred_colours, hatches, names,
               data.cohesion(wall, data.pred[wall]), REDUCE)
    axes[1].set_title(
        f"predicted  —  {len(data.pred[wall])} clusters, "
        f"recovery@{IOU_MATCH} = {recovery:.3f}", loc="left")

    missed = [label(m["gt_route"]) for m in data.matches[wall] if m["iou"] < IOU_MATCH]
    fig.suptitle(f"{wall}  ·  route segmentation, threshold {data.threshold:.2f}",
                 x=0.0, y=1.0, ha="left", va="bottom", fontsize=13, fontweight="bold")
    fig.text(0.0, -0.01,
             "Colour identifies a ground-truth route; each predicted cluster is drawn in the colour of the "
             "route it was matched to,\nso a colour that changes between panels is a segmentation error. "
             "Every hold is tagged with the route it belongs to and that hold's mean P(same route) against the "
             "rest of its own\ngroup — the quantity average linkage clusters on, so a low percentage marks a "
             "hold the model only weakly binds.\nIn the lower panel \"?\" is a cluster matching no "
             "ground-truth route.\nMissed routes: " + (", ".join(missed) if missed else "none"),
             ha="left", va="top", fontsize=8.5, color=INK_2)
    save(fig, f"w1_overlay_{wall.replace('.jpg', '')}")


def fig_contingency(data, wall):
    gt, pred = data.gt[wall], data.pred[wall]
    gt_match = data.gt_match(wall)
    matched = sorted(gt_match, key=lambda r: -gt_match[r][1])
    rows = matched + [r for r in data.gt_order(wall) if r not in gt_match]
    cols = [gt_match[r][0] for r in matched] + [k for k in sorted(pred) if k not in
                                                {gt_match[r][0] for r in matched}]

    counts = np.array([[len(gt[r] & pred[k]) for k in cols] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(1.0 + 0.46 * len(cols), 1.0 + 0.40 * len(rows)))
    image = ax.imshow(counts, cmap=SEQ_CMAP, vmin=0, vmax=counts.max(), zorder=2)

    high = 0.62 * counts.max()
    for i in range(len(rows)):
        for j in range(len(cols)):
            if counts[i, j]:
                ax.text(j, i, int(counts[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if counts[i, j] > high else INK, zorder=4)
    for i, route in enumerate(rows):
        if route in gt_match:
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, zorder=5,
                                   edgecolor=CRITICAL if gt_match[route][1] < IOU_MATCH else INK,
                                   linewidth=2.0))

    ax.set_xticks(range(len(cols)), [str(k) for k in cols], fontsize=8)
    ax.set_yticks(range(len(rows)), [label(r) for r in rows], fontsize=8)
    ax.set_xlabel("predicted cluster")
    ax.set_ylabel("ground-truth route")
    ax.set_xticks(np.arange(len(cols) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rows) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2.0)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    bar.set_label("holds in common", color=INK_2)
    bar.outline.set_visible(False)
    bar.ax.tick_params(color=MUTED, labelcolor=INK_2)

    ax.set_title(f"{wall}  —  route × cluster contingency, recovery@{IOU_MATCH} = "
                 f"{data.recovery(wall):.3f}", loc="left")
    fig.text(0.0, -0.02,
             "Rows ordered by match quality. A boxed cell is the assigned match (red if IoU < "
             f"{IOU_MATCH}); a row with no box\nshares no hold with any cluster left to assign it. "
             "Off-diagonal mass in a row is a split route, in a column a merge.",
             ha="left", va="top", fontsize=8.5, color=INK_2)
    save(fig, f"w3_contingency_{wall.replace('.jpg', '')}")


def fig_colour_space(data, wall):
    gt = data.gt[wall]
    colours = data.colour_of_gt(wall)
    gt_match = data.gt_match(wall)
    cluster_of = data.cluster_of_hold(wall)
    holds = data.holds

    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    ax.axhline(128, color=GRID, linewidth=0.8, zorder=1)
    ax.axvline(128, color=GRID, linewidth=0.8, zorder=1)

    n_wrong = 0
    anchors, texts = [], []
    for route in data.gt_order(wall):
        ids = [h for h in gt[route] if h in holds.index]
        rows = holds.loc[ids]
        target = gt_match.get(route, (None, 0.0))[0]
        wrong = np.array([cluster_of.get(h) != target for h in ids])
        n_wrong += int(wrong.sum())
        a, b = rows["med_a"].to_numpy(), rows["med_b"].to_numpy()
        ax.scatter(a[~wrong], b[~wrong], s=85, marker="o", zorder=3,
                   facecolor=[colours[route]] * int((~wrong).sum()),
                   edgecolor="white", linewidth=1.2)
        ax.scatter(a[wrong], b[wrong], s=125, marker="X", zorder=4,
                   facecolor=[colours[route]] * int(wrong.sum()),
                   edgecolor=INK, linewidth=1.2)
        anchors.append((float(np.median(a)), float(np.median(b))))
        texts.append(label(route))

    span_x = np.ptp(ax.get_xlim())
    span_y = np.ptp(ax.get_ylim())
    for (x, y), text in zip(spread(anchors, 0.075 * span_x, 0.040 * span_y), texts):
        ax.text(x, y, text, fontsize=9, fontweight="bold", color=INK, zorder=6,
                ha="center", va="center", path_effects=STROKE)

    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markersize=9, markerfacecolor=MUTED,
               markeredgecolor="white", label="clustered with its route"),
        Line2D([], [], marker="X", linestyle="none", markersize=10, markerfacecolor=MUTED,
               markeredgecolor=INK, label="landed in the wrong cluster"),
    ], loc="lower right", fontsize=8.5)

    ax.set_xlabel("a*  (green → red)")
    ax.set_ylabel("b*  (blue → yellow)")
    ax.set_aspect("equal")
    grid_on(ax, axis="both")
    ax.set_title(f"{wall}  —  hold colour by ground-truth route", loc="left")
    fig.text(0.0, -0.01,
             f"One marker per hold, positioned by its median Lab chroma and coloured by its true route; "
             f"labels sit at each route's median.\n{n_wrong} of "
             f"{sum(len(v) for v in gt.values())} holds landed in the wrong cluster. Routes that pile up in "
             "the same region here are the ones the affinity\nmodel cannot separate on colour, and it has "
             "little else to go on when they are also physically adjacent.",
             ha="left", va="top", fontsize=8.5, color=INK_2)
    save(fig, f"w4_colour_{wall.replace('.jpg', '')}")


def fig_sweep(data):
    sweep = pd.DataFrame(data.metrics["sweep"])
    t, mean, sd = sweep["threshold"], sweep["recovery_mean"], sweep["recovery_sd"]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.fill_between(t, mean - sd, mean + sd, color=SLOT[0], alpha=0.10, zorder=2,
                    linewidth=0)
    ax.plot(t, mean, color=SLOT[0], linewidth=2.0, marker="o", markersize=5,
            markeredgecolor="white", markeredgewidth=1.0, zorder=4,
            label=f"route recovery @{IOU_MATCH} IoU (macro, ±1 sd)")
    ax.plot(t, sweep["pair_f1"], color=SLOT[1], linewidth=2.0, marker="o", markersize=5,
            markeredgecolor="white", markeredgewidth=1.0, zorder=4, label="pair F1")

    for series, colour, name in ((mean, SLOT[0], "recovery"), (sweep["pair_f1"], SLOT[1], "pair F1")):
        best = int(np.argmax(series))
        ax.scatter([t[best]], [series[best]], s=150, facecolor="none", zorder=5,
                   edgecolor=colour, linewidth=2.0)
        ax.annotate(f"{name} peaks at {t[best]:.2f}\n{series[best]:.3f}",
                    (t[best], series[best]), textcoords="offset points",
                    xytext=(8, -26 if name == "recovery" else 12), fontsize=8.5, color=colour,
                    fontweight="bold")

    ax.set_xlabel("affinity threshold")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    grid_on(ax)
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_title("The pair-level and route-level optima are not the same threshold", loc="left")
    fig.text(0.0, -0.02,
             "Both curves are fractions on one axis. Tuning the threshold on pair F1 costs route recovery, "
             "because a merge or split\ncosts a whole route regardless of how few pairs caused it. "
             "Recovery is flat across 0.25–0.40, so the choice is not delicate.",
             ha="left", va="top", fontsize=8.5, color=INK_2)
    save(fig, "a1_threshold_sweep")


def fig_per_wall(data):
    recovery = data.metrics["per_wall_recovery"]
    order = sorted(recovery, key=lambda w: recovery[w])
    values = [recovery[w] for w in order]
    n_gt = [len(data.gt[w]) for w in order]
    mean = float(np.mean(values))

    fig, ax = plt.subplots(figsize=(7.4, 0.23 * len(order) + 1.6))
    ax.barh(range(len(order)), values, height=0.72, color=SLOT[0], zorder=3)
    ax.axvline(mean, color=INK_2, linewidth=1.4, linestyle="--", zorder=4)
    ax.annotate(f"macro mean {mean:.3f}", (mean, len(order) - 0.2),
                textcoords="offset points", xytext=(6, 0), fontsize=8.5,
                color=INK_2, fontweight="bold", va="center")
    for i, (value, n) in enumerate(zip(values, n_gt)):
        ax.text(value + 0.012, i, f"{value:.2f}   {n} routes", va="center",
                fontsize=8, color=INK_2)

    ax.set_yticks(range(len(order)), [w.replace(".jpg", "") for w in order], fontsize=8)
    ax.set_xlim(0, 1.18)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel(f"routes recovered @{IOU_MATCH} IoU")
    grid_on(ax, axis="x")
    ax.set_title("Per-wall recovery  —  the error concentrates on dense walls", loc="left")
    fig.text(0.0, -0.01,
             "Leave-one-wall-out. Each bar is one wall, annotated with its ground-truth route count. "
             "Twelve walls are perfect.\nRecovery falls as a wall carries more routes (Spearman ρ = −0.68, "
             "p < 1e-4): the ten walls with 12 or more routes\naverage 0.59 against 0.87 for the other "
             "twenty-four — though 34 and 21 show density alone is not destiny.",
             ha="left", va="top", fontsize=8.5, color=INK_2)
    save(fig, "a2_per_wall_recovery")


def fig_iou_ecdf(data):
    ious = np.sort([m["iou"] for ms in data.matches.values() for m in ms])
    y = np.arange(1, len(ious) + 1) / len(ious)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.step(ious, y, where="post", color=SLOT[0], linewidth=2.0, zorder=4)
    for mark in (0.5, IOU_MATCH):
        share = float((ious < mark).mean())
        ax.axvline(mark, color=MUTED, linewidth=1.2, linestyle="--", zorder=3)
        ax.annotate(f"{share:.1%} of routes\nbelow {mark}", (mark, 0.06),
                    textcoords="offset points", xytext=(-6, 0), ha="right",
                    fontsize=8.5, color=INK_2, fontweight="bold")

    ax.set_xlabel("IoU of the matched ground-truth route")
    ax.set_ylabel("cumulative share of routes")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    grid_on(ax, axis="both")
    ax.set_title("Routes are recovered exactly or not at all", loc="left")
    fig.text(0.0, -0.02,
             f"Empirical CDF over all {len(ious)} matched ground-truth routes. The curve is flat through the "
             "middle and jumps at 1.0:\nthere is almost no population of partially-recovered routes, which is "
             "why the 0.8 criterion is not a harsh one.",
             ha="left", va="top", fontsize=8.5, color=INK_2)
    save(fig, "a3_matched_iou_ecdf")


def fig_affinity(data):
    pairs = data.pairs
    same = pairs.loc[pairs["same_route"] == 1, "oof_probs"]
    diff = pairs.loc[pairs["same_route"] == 0, "oof_probs"]
    bins = np.linspace(0, 1, 61)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for values, colour, name in ((diff, SLOT[1], f"different route  (n={len(diff):,})"),
                                 (same, SLOT[0], f"same route  (n={len(same):,})")):
        ax.hist(values, bins=bins, histtype="step", color=colour, linewidth=2.0,
                zorder=3, label=name)
    ax.axvline(data.threshold, color=INK, linewidth=1.6, zorder=5)
    ax.annotate(f"clustering threshold {data.threshold:.2f}", (data.threshold, 0.60),
                xycoords=("data", "axes fraction"), textcoords="offset points",
                xytext=(8, 0), fontsize=8.5, color=INK, fontweight="bold")

    ax.set_yscale("log")
    ax.set_xlabel("out-of-fold P(same route)")
    ax.set_ylabel("pairs (log scale)")
    ax.set_xlim(0, 1)
    grid_on(ax)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title("Pairwise affinity separates cleanly, and the residue is what breaks routes", loc="left")
    fig.text(0.0, -0.02,
             "94% of different-route pairs score below 0.1 and half of same-route pairs above 0.9, so most "
             "pairs are easy. The log\naxis exposes what is left: 12,640 pairs sit in the ambiguous 0.2–0.8 "
             "band, and 1,723 are confident mistakes. Each one\nof those can merge or split a whole route, "
             "which is why 85% pair F1 becomes 78% route recovery.",
             ha="left", va="top", fontsize=8.5, color=INK_2)
    save(fig, "a4_affinity_separation")


def classify_misses(data, wall):
    gt, pred = data.gt[wall], data.pred[wall]
    gt_match = data.gt_match(wall)
    kinds = defaultdict(list)
    for route in data.gt_order(wall):
        if gt_match.get(route, (None, 0.0))[1] >= IOU_MATCH:
            continue
        touched = {k for k, holds in pred.items() if holds & gt[route]}
        foreign = any(pred[k] - gt[route] for k in touched)
        if len(touched) == 1 and foreign:
            kinds["merged"].append(route)
        elif len(touched) > 1 and not foreign:
            kinds["split"].append(route)
        else:
            kinds["mixed"].append(route)
    return kinds


def fig_taxonomy(data):
    kinds = ["merged", "split", "mixed"]
    per_wall = {w: classify_misses(data, w) for w in data.walls}
    walls = sorted((w for w in data.walls if any(per_wall[w].values())),
                   key=lambda w: -sum(len(v) for v in per_wall[w].values()))
    totals = {k: sum(len(per_wall[w][k]) for w in data.walls) for k in kinds}

    fig, ax = plt.subplots(figsize=(7.4, 0.26 * len(walls) + 2.2))
    left = np.zeros(len(walls))
    for kind, colour in zip(kinds, SLOT[:3]):
        values = np.array([len(per_wall[w][kind]) for w in walls], dtype=float)
        ax.barh(range(len(walls)), values, left=left, height=0.70, color=colour,
                edgecolor=SURFACE, linewidth=2.0, zorder=3,
                label=f"{kind}  ({totals[kind]})")
        left += values
    for i, total in enumerate(left):
        ax.text(total + 0.12, i, f"{int(total)}", va="center", fontsize=8, color=INK_2)

    ax.set_yticks(range(len(walls)), [w.replace(".jpg", "") for w in walls], fontsize=8)
    ax.set_xlabel("missed ground-truth routes")
    grid_on(ax, axis="x")
    ax.legend(loc="upper right", fontsize=8.5, title="failure mode", title_fontsize=8.5)
    ax.set_title(f"How the {int(left.sum())} missed routes fail", loc="left")
    fig.text(0.0, -0.01,
             "merged = the route's holds all landed in one cluster that also holds another route.   "
             "split = its holds were\ndivided across clusters that are otherwise clean.   "
             "mixed = both at once.\n82 of the 90 failures involve a merge and only 8 are pure splits, so "
             "the affinity model's dominant error is calling\ntwo different routes the same — sharper "
             "discrimination between adjacent look-alike routes is worth more than\nanything aimed at "
             "holding a single route together.",
             ha="left", va="top", fontsize=8.5, color=INK_2)
    save(fig, "a5_error_taxonomy")


def fig_linkage(data):
    thresholds = np.arange(0.20, 0.95, 0.05)
    linkages = ["single", "average", "complete"]
    names = {"single": "single  (= connected components)", "average": "average",
             "complete": "complete"}
    curves = Parallel(n_jobs=min(12, len(thresholds)))(
        delayed(macro_recovery)(data.pairs, t, linkage)
        for linkage in linkages for t in thresholds
    )
    curves = {linkage: np.array([c.mean() for c in curves[i * len(thresholds):(i + 1) * len(thresholds)]])
              for i, linkage in enumerate(linkages)}

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for linkage, colour in zip(linkages, SLOT[:3]):
        values = curves[linkage]
        ax.plot(thresholds, values, color=colour, linewidth=2.0, marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=1.0, zorder=4, label=names[linkage])
        best = int(np.argmax(values))
        ax.annotate(f"{values[best]:.3f}", (thresholds[best], values[best]),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=8.5, color=colour, fontweight="bold")

    ax.set_xlabel("affinity threshold")
    ax.set_ylabel(f"route recovery @{IOU_MATCH} IoU (macro)")
    ax.set_ylim(0, 1)
    grid_on(ax)
    ax.legend(loc="lower left", fontsize=8.5, title="linkage", title_fontsize=8.5)
    ax.set_title("Average linkage recovers 14 points more than connected components", loc="left")
    fig.text(0.0, -0.02,
             "Identical pairwise affinities, clustered three ways. Thresholding the affinity graph and taking "
             "connected components is\nexactly single linkage: its transitive chaining merges any two routes "
             "joined by one confident mistake, so it only becomes\nusable at thresholds above 0.7 where it "
             "finally stops chaining — and still peaks 14 points below average linkage.",
             ha="left", va="top", fontsize=8.5, color=INK_2)
    save(fig, "a8_linkage_ablation")
    return {k: float(v.max()) for k, v in curves.items()}


def hard_routes(data, wall):
    holds = data.holds
    wall_holds = holds[(holds["image_file"] == wall) &
                       (holds["category_id"] != VOLUME_CATEGORY_ID)]
    scale = float(np.median(np.sqrt(wall_holds["area"])))

    stats = {}
    for route, ids in data.gt[wall].items():
        rows = holds.loc[[h for h in ids if h in holds.index]]
        ab = rows[["med_a", "med_b"]].to_numpy()
        xy = rows[["cx", "cy"]].to_numpy() / scale
        centroid = np.median(ab, axis=0)
        spread = float(np.percentile(np.linalg.norm(ab - centroid, axis=1), 75))
        stats[route] = (centroid, max(spread, HARD_COLOUR_FLOOR), xy)

    hard = set()
    for route, (centroid, tol, xy) in stats.items():
        for other, (other_centroid, _, other_xy) in stats.items():
            if other == route:
                continue
            if np.linalg.norm(centroid - other_centroid) > tol:
                continue
            gap = np.linalg.norm(xy[:, None, :] - other_xy[None, :, :], axis=2).min()
            if gap <= HARD_SPATIAL:
                hard.add(route)
                break
    return hard


def fig_hard_easy(data):
    groups = {"colour-confusable neighbour\nwithin 6 hold-units": [], "no such neighbour": []}
    keys = list(groups)
    for wall in data.walls:
        hard = hard_routes(data, wall)
        gt_match = data.gt_match(wall)
        for route in data.gt_order(wall):
            hit = gt_match.get(route, (None, 0.0))[1] >= IOU_MATCH
            groups[keys[0] if route in hard else keys[1]].append(hit)

    shares = [float(np.mean(groups[k])) for k in keys]
    counts = [len(groups[k]) for k in keys]
    overall = float(np.mean(sum(groups.values(), [])))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.bar(range(2), shares, width=0.5, color=SLOT[0], zorder=3)
    ax.axhline(overall, color=INK_2, linewidth=1.4, linestyle="--", zorder=4)
    ax.annotate(f"all routes  {overall:.3f}", (0.5, overall), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=8.5, color=INK_2, fontweight="bold")
    for i, (share, n) in enumerate(zip(shares, counts)):
        ax.text(i, share + 0.02, f"{share:.3f}\n{n} routes  ({n / sum(counts):.0%})",
                ha="center", fontsize=9, color=INK, fontweight="bold")

    ax.set_xticks(range(2), keys, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(f"routes recovered @{IOU_MATCH} IoU")
    grid_on(ax)
    ax.set_title("Most of the error lives in the sixth of routes with a confusable neighbour",
                 loc="left")
    fig.text(0.0, -0.06,
             "A route is confusable when another route on the same wall sits within its own p75 colour spread "
             "(floored at 6 Lab units)\nand has a hold within 6 hold-units. Micro-averaged over routes, not "
             "walls. The easy majority is close to solved;\nthe thesis-relevant work is the hard minority.",
             ha="left", va="top", fontsize=8.5, color=INK_2)
    save(fig, "a9_hard_vs_easy")
    return dict(zip(["hard", "easy"], zip(shares, counts)))


def fig_importance(data):
    importance = data.metrics["importance"]
    total = sum(importance.values())
    order = sorted(importance, key=lambda f: importance[f])
    shares = [importance[f] / total for f in order]

    fig, ax = plt.subplots(figsize=(7.0, 0.26 * len(order) + 1.6))
    ax.barh(range(len(order)), shares, height=0.72, color=SLOT[0], zorder=3)
    for i, share in enumerate(shares):
        ax.text(share + 0.004, i, f"{share:.1%}", va="center", fontsize=8, color=INK_2)

    ax.set_yticks(range(len(order)), order, fontsize=8.5)
    ax.set_xlim(0, max(shares) * 1.18)
    ax.set_xlabel("share of total gain")
    grid_on(ax, axis="x")
    ax.set_title("What the pairwise scorer actually uses", loc="left")
    fig.text(0.0, -0.01,
             "Mean XGBoost gain across the 34 leave-one-wall-out folds. Colour difference dominates, but the "
             "horizontal separation\nbetween holds (dx) outranks both plain distance and dy — routes run "
             "vertically and are separated horizontally.",
             ha="left", va="top", fontsize=8.5, color=INK_2)
    save(fig, "a6_feature_importance")


def fig_pr_curve(data):
    pairs = data.pairs
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for wall, group in pairs.groupby("image_file"):
        if group["same_route"].nunique() < 2:
            continue
        precision, recall, _ = precision_recall_curve(group["same_route"], group["oof_probs"])
        ax.plot(recall, precision, color=MUTED, linewidth=0.9, alpha=0.45, zorder=3)
    ax.plot([], [], color=MUTED, linewidth=0.9, alpha=0.8, label="individual walls")

    precision, recall, thresholds = precision_recall_curve(pairs["same_route"], pairs["oof_probs"])
    ap = average_precision_score(pairs["same_route"], pairs["oof_probs"])
    ax.plot(recall, precision, color=SLOT[0], linewidth=2.2, zorder=5,
            label=f"pooled  (AP = {ap:.3f})")

    k = int(np.searchsorted(thresholds, data.threshold))
    ax.scatter([recall[k]], [precision[k]], s=150, facecolor="none", edgecolor=SLOT[1],
               linewidth=2.0, zorder=6)
    ax.annotate(f"operating point {data.threshold:.2f}\nP {precision[k]:.2f}  R {recall[k]:.2f}",
                (recall[k], precision[k]), textcoords="offset points", xytext=(-10, -34),
                ha="right", fontsize=8.5, color=SLOT[1], fontweight="bold")

    ax.axhline(pairs["same_route"].mean(), color=MUTED, linestyle=":", linewidth=1.2, zorder=2)
    ax.annotate(f"chance {pairs['same_route'].mean():.3f}", (0.52, pairs["same_route"].mean()),
                textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8.5, color=MUTED)

    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    grid_on(ax, axis="both")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_title("Pairwise same-route classification, leave-one-wall-out", loc="left")
    fig.text(0.0, -0.02,
             "Every wall is held out of training entirely, so each grey curve is a wall the model has never "
             "seen. Pooled AP is 0.914,\nbut a handful of walls fall far below it, and those are the same "
             "walls that lose routes: per-wall AP and per-wall\nrecovery correlate at Spearman ρ = 0.78 "
             "(p < 1e-6). Affinity quality, not the clustering step, is the binding constraint.",
             ha="left", va="top", fontsize=8.5, color=INK_2)
    save(fig, "a7_pr_curve")


def main():
    data = Data()
    os.makedirs(FIG_DIR, exist_ok=True)
    print("aggregate")
    fig_sweep(data)
    fig_per_wall(data)
    fig_iou_ecdf(data)
    fig_affinity(data)
    fig_taxonomy(data)
    fig_importance(data)
    fig_pr_curve(data)
    peaks = fig_linkage(data)
    hard = fig_hard_easy(data)

    print("per wall")
    for wall in WALLS:
        fig_overlay(data, wall)
        fig_contingency(data, wall)
        fig_colour_space(data, wall)

    print("\nlinkage peaks:", {k: round(v, 4) for k, v in peaks.items()})
    print("hard/easy:", {k: (round(s, 4), n) for k, (s, n) in hard.items()})


if __name__ == "__main__":
    main()
