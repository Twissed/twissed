"""matplotlib.py file

    Python package for beam dynamics analysis in laser-plasma acceleration
    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 21/07/2023
"""


try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib.colors as colors
    from matplotlib import cm, patches
    from matplotlib.colors import ListedColormap
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
except ImportError:
    print("WARNING: matplotlib package not found")

    # Dummy class
    class plt:
        figure = None
        Axes = None


try:
    import seaborn as sns
except ImportError:
    print("WARNING: seaborn package not found")

    # Dummy class
    class sns:
        pass
