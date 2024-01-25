"""colormap.py file

    Python package for beam dynamics analysis in laser-plasma acceleration
    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 21/07/2023
"""

import numpy as np

# twissed
from .matplotlib import plt, ListedColormap, sns


cm_list = [
    "Accent",
    "Accent_r",
    "Blues",
    "Blues_r",
    "BrBG",
    "BrBG_r",
    "BuGn",
    "BuGn_r",
    "BuPu",
    "BuPu_r",
    "CMRmap",
    "CMRmap_r",
    "Dark2",
    "Dark2_r",
    "GnBu",
    "GnBu_r",
    "Greens",
    "Greens_r",
    "Greys",
    "Greys_r",
    "OrRd",
    "OrRd_r",
    "Oranges",
    "Oranges_r",
    "PRGn",
    "PRGn_r",
    "Paired",
    "Paired_r",
    "Pastel1",
    "Pastel1_r",
    "Pastel2",
    "Pastel2_r",
    "PiYG",
    "PiYG_r",
    "PuBu",
    "PuBuGn",
    "PuBuGn_r",
    "PuBu_r",
    "PuOr",
    "PuOr_r",
    "PuRd",
    "PuRd_r",
    "Purples",
    "Purples_r",
    "RdBu",
    "RdBu_r",
    "RdGy",
    "RdGy_r",
    "RdPu",
    "RdPu_r",
    "RdYlBu",
    "RdYlBu_r",
    "RdYlGn",
    "RdYlGn_r",
    "Reds",
    "Reds_r",
    "Set1",
    "Set1_r",
    "Set2",
    "Set2_r",
    "Set3",
    "Set3_r",
    "Spectral",
    "Spectral_r",
    "Wistia",
    "Wistia_r",
    "YlGn",
    "YlGnBu",
    "YlGnBu_r",
    "YlGn_r",
    "YlOrBr",
    "YlOrBr_r",
    "YlOrRd",
    "YlOrRd_r",
    "afmhot",
    "afmhot_r",
    "autumn",
    "autumn_r",
    "binary",
    "binary_r",
    "bone",
    "bone_r",
    "brg",
    "brg_r",
    "bwr",
    "bwr_r",
    "cividis",
    "cividis_r",
    "cool",
    "cool_r",
    "coolwarm",
    "coolwarm_r",
    "copper",
    "copper_r",
    "cubehelix",
    "cubehelix_r",
    "flag",
    "flag_r",
    "gist_earth",
    "gist_earth_r",
    "gist_gray",
    "gist_gray_r",
    "gist_heat",
    "gist_heat_r",
    "gist_ncar",
    "gist_ncar_r",
    "gist_rainbow",
    "gist_rainbow_r",
    "gist_stern",
    "gist_stern_r",
    "gist_yarg",
    "gist_yarg_r",
    "gnuplot",
    "gnuplot2",
    "gnuplot2_r",
    "gnuplot_r",
    "gray",
    "gray_r",
    "hot",
    "hot_r",
    "hsv",
    "hsv_r",
    "icefire",
    "icefire_r",
    "inferno",
    "inferno_r",
    "magma",
    "magma_r",
    "mako",
    "mako_r",
    "nipy_spectral",
    "nipy_spectral_r",
    "ocean",
    "ocean_r",
    "pink",
    "pink_r",
    "plasma",
    "plasma_r",
    "prism",
    "prism_r",
    "rainbow",
    "rainbow_r",
    "rocket",
    "rocket_r",
    "seismic",
    "seismic_r",
    "spring",
    "spring_r",
    "summer",
    "summer_r",
    "tab10",
    "tab10_r",
    "tab20",
    "tab20_r",
    "tab20b",
    "tab20b_r",
    "tab20c",
    "tab20c_r",
    "terrain",
    "terrain_r",
    "twilight",
    "twilight_r",
    "twilight_shifted",
    "twilight_shifted_r",
    "viridis",
    "viridis_r",
    "vlag",
    "vlag_r",
    "winter",
    "winter_r",
    "flare",
    "flare_r",
    "crest",
    "crest_r",
]


class Cmap:
    def __init__(self):
        for name in cm_list:
            self.__dict__[name] = sns.color_palette(name, as_cmap=True)

        self.light_seagreen = sns.color_palette("light:seagreen", as_cmap=True)
        self.dark_seagreen = sns.color_palette("dark:seagreen", as_cmap=True)

        BLACK = [0.0, 0.0, 0.0, 1.0]
        WHITE = [1.0, 1.0, 1.0, 1.0]
        PURPLE = [127 / 256, 0.0, 1.0, 1.0]
        BLUE = [0.0, 0.0, 1.0, 1.0]
        CYAN = [0.0, 1.0, 1.0, 1.0]
        GREEN = [0.0, 1.0, 0.0, 1.0]
        YELLOW = [1.0, 1.0, 0.0, 1.0]
        RED = [1.0, 0.0, 0.0, 1.0]
        RED_CEA = [229 / 256, 0.0, 25 / 256, 1.0]

        step1 = np.linspace(WHITE, PURPLE, 16)
        step2 = np.linspace(PURPLE, BLUE, 40)
        step3 = np.linspace(BLUE, CYAN, 40)
        step4 = np.linspace(CYAN, GREEN, 40)
        step5 = np.linspace(GREEN, YELLOW, 40)
        step6 = np.linspace(YELLOW, RED, 40)
        step7 = np.linspace(RED, BLACK, 40)

        tracewin = np.concatenate(
            (step1, step2, step3, step4, step5, step6, step7), axis=0
        )
        self.tracewin = ListedColormap(tracewin)
        self.tracewin_r = ListedColormap(tracewin[-1:0:-1])

        step1 = np.linspace(WHITE, PURPLE, 16)
        step2 = np.linspace(PURPLE, BLUE, 48)
        step3 = np.linspace(BLUE, CYAN, 48)
        step4 = np.linspace(CYAN, GREEN, 48)
        step5 = np.linspace(GREEN, YELLOW, 48)
        step6 = np.linspace(YELLOW, RED, 48)

        tracewin_light = np.concatenate(
            (step1, step2, step3, step4, step5, step6), axis=0
        )
        self.tracewin_light = ListedColormap(tracewin_light)
        self.tracewin_light_r = ListedColormap(tracewin_light[-1:0:-1])

        red_cea = np.linspace([1, 1, 1, 1], RED_CEA, 256)
        self.red_cea = ListedColormap(red_cea)
        self.red_cea_r = ListedColormap(red_cea[-1:0:-1])

        red_cea_dark = np.linspace([0, 0, 0, 1], RED_CEA, 256)
        self.red_cea_dark = ListedColormap(red_cea_dark)
        self.red_cea_dark_r = ListedColormap(red_cea_dark[-1:0:-1])

        LIGHTBLUE_ICED = [0.4455597, 0.6170920, 0.7156208, 1.0]
        BLUE_ICED = [0.36802290348805533, 0.4411620102344578, 0.6229388227089065, 1.0]
        DARKBLUE_ICED = [0.19219, 0.11144, 0.23278, 1.0]

        step1 = np.linspace(WHITE, LIGHTBLUE_ICED, 86)
        step2 = np.linspace(LIGHTBLUE_ICED, BLUE_ICED, 85)
        step3 = np.linspace(BLUE_ICED, DARKBLUE_ICED, 85)

        sky_iced = np.concatenate((step1, step2, step3), axis=0)
        self.sky_iced = ListedColormap(sky_iced)
        self.sky_iced_r = ListedColormap(sky_iced[-1:0:-1])

        ICEFIRE = [
            WHITE,
            [0.72888063, 0.89639109, 0.85488394, 1.0],
            [0.51728854, 0.75509528, 0.81194156, 1.0],
            [0.29623491, 0.61072284, 0.80569021, 1.0],
            [0.23176013, 0.44119137, 0.80494325, 1.0],
            [0.28233561, 0.28527482, 0.58742866, 1.0],
            [0.19619947, 0.18972425, 0.31383846, 1.0],
            [0.12586516, 0.12363617, 0.1448459, 1.0],
            [0.18468769, 0.12114722, 0.13306426, 1.0],
            [0.36178937, 0.1589124, 0.20807639, 1.0],
            [0.58932081, 0.18117827, 0.26800409, 1.0],
            [0.79819286, 0.25427223, 0.22352658, 1.0],
            [0.91830723, 0.44790913, 0.21916352, 1.0],
            [0.96809871, 0.66971662, 0.45830232, 1.0],
            [0.9992197, 0.83100723, 0.6764127, 1.0],
        ]

        sky_icefire = np.linspace(ICEFIRE[0], ICEFIRE[1], 20)
        for i in range(1, len(ICEFIRE) - 1):
            sky_icefire = np.concatenate(
                (sky_icefire, np.linspace(ICEFIRE[i], ICEFIRE[i + 1], 20)), axis=0
            )
        self.sky_icefire = ListedColormap(sky_icefire)
        self.sky_icefire_r = ListedColormap(sky_icefire[-1:0:-1])

        EARLI_yellow = [217 / 256, 183 / 256, 41 / 256, 1.0]
        EARLI_blue = [51 / 256, 86 / 256, 166 / 256, 1.0]

        cm_earli_yellow = np.linspace([1, 1, 1, 1], EARLI_yellow, 256)
        self.cm_earli_yellow = ListedColormap(cm_earli_yellow)
        self.cm_earli_yellow_r = ListedColormap(cm_earli_yellow[-1:0:-1])

        cm_earli_blue = np.linspace([1, 1, 1, 1], EARLI_blue, 256)
        self.cm_earli_blue = ListedColormap(cm_earli_blue)
        self.cm_earli_blue_r = ListedColormap(cm_earli_blue[-1:0:-1])

        step1 = np.linspace([1, 1, 1, 1], EARLI_blue, 75)
        step2 = np.linspace(EARLI_blue, EARLI_yellow, 181)
        step3 = np.linspace(EARLI_yellow, [233 / 256, 211 / 256, 126 / 256, 1], 20)
        cm_earli_blueyellow = np.concatenate((step1, step2, step3), axis=0)
        self.cm_earli_blueyellow = ListedColormap(cm_earli_blueyellow)
        self.cm_earli_blueyellow_r = ListedColormap(cm_earli_blueyellow[-1:0:-1])
