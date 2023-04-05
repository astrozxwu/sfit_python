# NOTE: python implementation of sfit.f
import argparse
import os

import numpy as np
import toml

from utils import flux2mag, gbad, mag2flux, plx_config, vgetb0p

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', help='config file')
args = parser.parse_args()
config = args.config

fitconfig = toml.load(config)
RA = fitconfig["RA"]
Dec = fitconfig["Dec"]
t0par = fitconfig["t0par"]
getavec = plx_config(t0par, RA, Dec)
interactive = fitconfig["interactive"]
verbose = fitconfig["verbose"]

parameters = ["t0", "u0", "tE", "rhos", "piEN", "piEE"]
parameters_to_fit = fitconfig["parameters"]["to_fit"]
parameters_fixed = fitconfig["parameters"]["fixed"]

datadir = fitconfig["datadir"]
outdir = fitconfig["outdir"]
photfiles = fitconfig["phot"]
nobs = len(photfiles)
nparam = 6 + 2 * nobs

fmt = nparam * "{:10.5f}"
fmt_str = 6 * "{:>10s}" + 2 * nobs * "{:>10s}"
fmt_chi2 = nobs * "{:10.1f}"
fmt_knt = nobs * "{:10.0f}"

fixed = np.zeros(nparam)
for i in range(len(parameters)):
    if parameters[i] in parameters_fixed:
        fixed[i] = 1

fixed = fixed == 1


# NOTE: load data
flux = []
for idx, phot in enumerate(photfiles):
    if phot["name"].endswith(".pysis"):
        usecols = (0, 3, 4)
    else:
        usecols = (0, 1, 2)
    df = np.loadtxt(os.path.join(datadir, phot["name"]), usecols=usecols)
    if "escale" in phot.keys():
        df[:, 2] = df[:, 2] * phot["escale"]
    if "bad" in phot.keys():
        # TODO: mask some data
        mask = np.full(df.shape, False)
        mask[phot["bad"], :] = True
        df = np.ma.asarray(df)
        df.mask = mask
    if "diff" in phot.keys():
        # TODO: for difference photometry
        continue
    else:
        _flux = mag2flux(df)
    flux.append(_flux)
knt = [len(i[0]) for i in flux]


# NOTE: compute and return the next step
def next_step(param, step_size):
    d = np.zeros(nparam)
    b = np.zeros((nparam, nparam))
    chi2 = np.zeros(nobs)
    for nob, _flux in enumerate(flux):
        t, fl, ferr = _flux
        t0, u0, tE, rhos, piEN, piEE = param[:6]
        tau = (t - t0) / tE
        qn, qe = getavec(t)

        dtau = piEN * qn + piEE * qe
        du0 = -piEN * qe + piEE * qn
        taup = tau + dtau
        u0p = u0 - du0
        u2 = taup**2 + u0p**2
        u = np.sqrt(u2)
        z = u / rhos
        b0, b1, db0, db1 = vgetb0p(z)
        amp = (u2 + 2) / u / np.sqrt(u2 + 4)
        amp_fs = amp * b0

        # NOTE: some derivatives...
        dampdu = -8 * amp / u / (u2 + 2) / (u2 + 4)
        dudu0p = u0p / u
        dudtaup = taup / u
        dtaupdt0 = -1 / tE
        dtaupdtE = -tau / tE
        dampfsdu = dampdu * b0 + amp * db0 / rhos
        fs, fb = param[2 * nob + 6], param[2 * nob + 7]
        grad = np.zeros((nparam, len(t)))
        grad[0] = dampfsdu * dudtaup * dtaupdt0 * fs
        grad[1] = dampfsdu * dudu0p * fs
        grad[2] = dampfsdu * dudtaup * dtaupdtE * fs
        grad[3] = -amp * db0 * u / rhos**2 * fs
        grad[4] = dampfsdu * (dudtaup * qn + dudu0p * qe) * fs
        grad[5] = dampfsdu * (dudtaup * qe - dudu0p * qn) * fs
        grad[6 + 2 * nob] = amp_fs
        grad[7 + 2 * nob] = 1

        y = (fl - amp_fs * fs - fb) / ferr
        grad = grad / ferr
        d += np.matmul(grad, y)
        b += np.matmul(grad, grad.T)
        chi2[nob] = np.sum(y**2)

    b[fixed, fixed] = 1e16
    da = np.linalg.solve(b, d)
    param_pre = param + da * step_size
    if verbose:
        print()
        print(fmt_str.format(*(parameters + nobs * ["fs", "fb"])))
        print(fmt.format(*param))
        print(fmt.format(*da))
        print(fmt.format(*param_pre))
        print()
        print("{:>10.1f}".format(np.sum(chi2)), fmt_chi2.format(*chi2))
        print("{:>10.0f}".format(np.sum(knt)), fmt_knt.format(*knt))
    return param_pre, chi2, da, b


def generate_lc(param, ref=0):
    fs_ref, fb_ref = param[2 * ref + 6], param[2 * ref + 7]
    masks = []
    bad = open(os.path.join(outdir, 'fort.81'), 'w')
    for nob, _flux in enumerate(flux):
        t, fl, ferr = _flux
        t0, u0, tE, rhos, piEN, piEE = param[:6]
        tau = (t - t0) / tE
        qn, qe = getavec(t)

        dtau = piEN * qn + piEE * qe
        du0 = -piEN * qe + piEE * qn
        taup = tau + dtau
        u0p = u0 - du0
        u2 = taup**2 + u0p**2
        u = np.sqrt(u2)
        z = u / rhos
        b0, b1, db0, db1 = vgetb0p(z)
        amp = (u2 + 2) / u / np.sqrt(u2 + 4)
        amp_fs = amp * b0
        fs, fb = param[2 * nob + 6], param[2 * nob + 7]
        f_align = (fl - fb) / fs * fs_ref + fb_ref
        ferr_align = ferr / fs * fs_ref
        data_aligned = flux2mag([t, f_align, ferr_align])
        f_model = amp_fs * fs_ref + fb_ref
        mag_model = flux2mag([f_model])
        output = np.concatenate([data_aligned, [data_aligned[1] - mag_model, amp_fs]]).T

        mask = np.full(fl.shape, False)
        mask[fitconfig["phot"][nob]["bad"]] = True

        np.savetxt(os.path.join(outdir, f"fort.{37+nob}"), output[~mask], fmt="%12.4f %8.3f %8.3f %8.3f %8.3f")

        chi2list = (fl - amp_fs * fs - fb) ** 2 / ferr**2
        mask = gbad(_flux, chi2list)
        print(mask)
        print(mask, file=bad)
    bad.close()
    t = np.concatenate(
        [
            np.linspace(t0 - 3 * tE, t0 - 0.5 * tE, 500),
            np.linspace(t0 - 0.5 * tE, t0 + 0.5 * tE, 1000)[1:-1],
            np.linspace(t0 + 0.5 * tE, t0 + 3 * tE, 500),
        ]
    )
    t0, u0, tE, rhos, piEN, piEE = param[:6]
    tau = (t - t0) / tE
    qn, qe = getavec(t)

    dtau = piEN * qn + piEE * qe
    du0 = -piEN * qe + piEE * qn
    taup = tau + dtau
    u0p = u0 - du0
    u2 = taup**2 + u0p**2
    u = np.sqrt(u2)
    z = u / rhos
    b0, b1, db0, db1 = vgetb0p(z)
    amp = (u2 + 2) / u / np.sqrt(u2 + 4)
    amp_fs = amp * b0
    f_model = amp_fs * fs_ref + fb_ref
    output = np.array([t, flux2mag([f_model]), amp_fs]).T
    np.savetxt(os.path.join(outdir, "fort.35"), output, fmt="%12.4f %8.4f %8.4f")

    return masks


if __name__ == "__main__":
    param_lens = {**parameters_fixed, **parameters_to_fit}
    param_lens = [param_lens[i] for i in parameters]
    param_phot = [1.0, 0] * nobs
    for i in range(nobs):
        param_phot[2 * i] = fitconfig["parameters"]["fs"][i]
        param_phot[2 * i + 1] = fitconfig["parameters"]["fb"][i]
    param = param_lens + param_phot
    if interactive:
        a = float(input())
        step_size = 0.1
        while a > 0:
            if a < 1:
                step_size = a
                n_step = 1
            elif a >= 1:
                n_step = int(a)
            else:
                break
            for i in range(n_step):
                param, chi2, _, _ = next_step(param, step_size)
            try:
                a = float(input())
            except Exception as e:
                break
    else:
        step_size = fitconfig["step_size"]
        n_step = fitconfig["n_step"]
        for _step_size, _n_step in zip(step_size, n_step):
            for i in range(_n_step):
                param, chi2, _, _ = next_step(param, _step_size)
    param_dict = dict(zip(parameters, param[: len(parameters)]))
    if verbose:
        param, chi2, _, b = next_step(param, 0.0001)
        ref = 0
        generate_lc(param, ref=0)
        np.savetxt(os.path.join(outdir, "best.par"), param[:8], fmt="%10.4f")
        err = np.linalg.inv(b)
        print()
        print(u'\u2500' * 10 * nparam)
        print("{:>10s}".format("output"))
        print(fmt.format(*param))
        print("{:>10s}".format("chi2"))
        print("{:>10.1f}".format(np.sum(chi2)), fmt_chi2.format(*chi2))
        print("{:>10.0f}".format(np.sum(knt)), fmt_knt.format(*knt))
        print(u'\u2500' * 10 * nparam)
        print("[parameters]")
        fs = ", ".join([f"{param[6 + 2*i]:8.3f}" for i in range(nobs)])
        fb = ", ".join([f"{param[7 + 2*i]:8.3f}" for i in range(nobs)])
        print(f"fs = [{fs}]")
        print(f"fb = [{fb}]")
        print()
        print("[parameters.to_fit]")
        for i, kwd in enumerate(fitconfig["parameters"]["to_fit"].keys()):
            print(f"{kwd} = {param_dict[kwd]:10.4f} # +/- {err[i,i]**0.5:.4f}")
        print()
        print("[parameters.fixed]")
        for i, kwd in enumerate(fitconfig["parameters"]["fixed"].keys()):
            print(f"{kwd} = {param_dict[kwd]:.4f}")
