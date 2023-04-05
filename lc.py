import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import toml
from astropy.time import Time
from matplotlib.offsetbox import AnchoredText

from utils import flux2mag

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', help='config file')
args = parser.parse_args()
config = args.config

plt.rcParams["font.size"] = "16"

clist = ['r', 'blue', 'lime', 'darkgreen', 'orange', 'magenta', 'lime', 'cyan', 'orliverab']
md, prefix = 35, 37

fitconfig = toml.load(config)

event = fitconfig["event"]
nobs = len(fitconfig["phot"])
taglist = [
    fitconfig["phot"][i]["name"].split(".")[0].replace(event, "").replace("-", " ").replace("_", " ")
    for i in range(nobs)
]

order = range(nobs)
res = 0.15
t0, u0, te, rhos, pien, piee, fs, fb = np.loadtxt(os.path.join(fitconfig["outdir"], "./best.par"))
Is = flux2mag([fs])
Is0 = Is - fitconfig["A_I"]
Ks0 = Is0 - 1.4 if Is0 < 16.5 else Is0 - 1
Kest = Ks0 + fitconfig["A_I"] / 7
t_ref = Time.now().jd1 - 2450000


model = np.loadtxt(os.path.join(fitconfig["outdir"], f'fort.{md}')).T
data = [np.loadtxt(os.path.join(fitconfig["outdir"], f'fort.{prefix+i}')).T for i in range(nobs)]

fig = plt.figure(figsize=(7, 8))
gs = fig.add_gridspec(2, 1, wspace=0, height_ratios=[3, 1])
(ax1, ax2) = gs.subplots(sharex=True, sharey=False)
ax1.get_xaxis().set_visible(False)
ax1.plot(model[0], model[1], linewidth=1, c='black', zorder=110)
ax2.plot(model[0], [0] * len(model[0]), linewidth=1, c='black', zorder=110)

for i in range(nobs):
    ax1.errorbar(
        data[i][0],
        data[i][1],
        yerr=data[i][2],
        fmt='o',
        markersize=6,
        label=taglist[i],
        fillstyle='none',
        markeredgewidth=1.5,
        c=clist[i],
        zorder=100 - order[i],
    )
    ax2.errorbar(
        data[i][0],
        data[i][3],
        yerr=data[i][2],
        fmt='o',
        markersize=6,
        label=taglist[i],
        fillstyle='none',
        markeredgewidth=1.5,
        c=clist[i],
        zorder=100 - order[i],
    )
A_now = np.interp(t_ref, model[0], model[2])
K_now = Kest - 2.5 * np.log10(A_now)
ax1.axvline(t_ref, color='r', label=f"K_now={K_now:.2f}\nA_now={A_now:.2f}")
ax2.set_ylim([-res, res])
ax2.set_xlim([t0 - 2 * te, t0 + 1 * te])
ax1.set_ylim([min(model[1]) - 0.1, max(model[1]) + 0.3])
s = f"$t_0={t0:.1f}\,\,u_0={u0:.2f}\,\,t_E={te:.1f}$"
at = AnchoredText(s, frameon=True, loc="lower left", zorder=120)
ax1.add_artist(at)
ax1.invert_yaxis()
ax2.invert_yaxis()
ax1.legend(loc="upper left")
ax1.set_ylabel('mag')
ax2.set_ylabel('residuals')
ax2.set_xlabel('HJD - 2450000')
# fig.align_xlabels()
# fig.align_ylabels()
fig.tight_layout()
plt.savefig(os.path.join(fitconfig["outdir"], f'LC.png'), bbox_inches='tight')
plt.show()
