import treecorr
import numpy as np


class rho_leakage:
    def __init__(
        self,
        ra_gal=None,
        dec_gal=None,
        g1_gal=None,
        g2_gal=None,
        w_gal=None,
        ra_star=None,
        dec_star=None,
        x_star=None,
        y_star=None,
        g1_star=None,
        g2_star=None,
        T_star=None,
        w_star=None,
        g1_psf=None,
        g2_psf=None,
        T_psf=None,
        w_psf=None,
        min_sep=0.1,
        max_sep=250,
        nbins=20,
        npatch=100,
        var_method="jackknife",
        rho_xy=False,
        seed=None,
    ):

        self._gal_cat = {
            "ra": ra_gal,
            "dec": dec_gal,
            "g1": g1_gal,
            "g2": g2_gal,
            "w": w_gal,
        }

        if rho_xy:
            self._star_cat = {
                "x": x_star,
                "y": y_star,
                "g1": g1_star,
                "g2": g2_star,
                "T": T_star,
                "w": w_star,
            }

            self.TC_config = {
                "max_sep": max_sep,
                "min_sep": min_sep,
                "nbins": nbins,
                "bin_slop": 0.1,
                "cross_patch_weight": "match",
            }
        else:
            self._star_cat = {
                "ra": ra_star,
                "dec": dec_star,
                "g1": g1_star,
                "g2": g2_star,
                "T": T_star,
                "w": w_star,
            }

            self.TC_config = {
                "ra_units": "degrees",
                "dec_units": "degrees",
                "max_sep": max_sep,
                "min_sep": min_sep,
                "sep_units": "arcmin",
                "nbins": nbins,
                "bin_slop": 0.1,
                "cross_patch_weight": "match",
            }

        self._psf_cat = {
            "g1": g1_psf,
            "g2": g2_psf,
            "T": T_psf,
            "w": w_psf,
        }

        self._npatch = npatch
        self.var_method = var_method
        self.rng = np.random.RandomState(seed)

    def set_rho_catalog(self):

        ra = self._star_cat["ra"]
        dec = self._star_cat["dec"]
        T_star = self._star_cat["T"]
        g1_star = self._star_cat["g1"]
        g2_star = self._star_cat["g2"]
        w_star = self._star_cat["w"]
        g1_psf = self._psf_cat["g1"]
        g2_psf = self._psf_cat["g2"]
        w_psf = self._psf_cat["w"]
        T_psf = self._psf_cat["T"]

        # shape_std_max = 5.
        # # Outlier rejection based on the size
        # R2_thresh_star = shape_std_max * np.std(T_star[w_star == 1]) + np.mean(
        #     T_star[w_star == 1])
        # R2_thresh_psf = shape_std_max * np.std(T_psf[w_psf == 1]) + np.mean(
        #     T_psf[w_psf == 1])
        # good_stars = (np.abs(T_star) < R2_thresh_star) \
        #     & (np.abs(T_psf) < R2_thresh_psf)

        ra = ra
        dec = dec
        g1 = g1_psf
        g2 = g2_psf
        dT = (T_star - T_psf) / T_star
        dg1 = g1_star - g1_psf
        dg2 = g2_star - g2_psf
        w1 = dT * g1_star
        w2 = dT * g2_star
        w = w_star

        w_tot = w_psf * w_star

        print("mean e = ", np.nanmean(g1), np.nanmean(g2))
        print("std e = ", np.nanstd(g1), np.nanstd(g2))
        print("mean T = ", np.nanmean(T_star))
        print("std T = ", np.nanstd(T_star))
        print("mean de = ", np.nanmean(dg1), np.nanmean(dg2))
        print("std de = ", np.nanstd(dg1), np.nanstd(dg2))
        print("mean dT = ", np.nanmean(T_star - T_psf))
        print("std dT = ", np.nanstd(T_star - T_psf))
        print("mean dT/T = ", np.nanmean(dT))
        print("std dT/T = ", np.nanstd(dT))

        # Substract mean
        if False:
            g1 -= np.mean(g1)
            g2 -= np.mean(g2)
            dg1 -= np.mean(dg1)
            dg2 -= np.mean(dg2)
            dT -= np.mean(dT)
            w1 -= np.mean(w1)
            w2 -= np.mean(w2)

        self.p_tc_cat = treecorr.Catalog(
            ra=ra,
            dec=dec,
            g1=g1,
            g2=g2,
            w=w_tot,
            config=self.TC_config,
            npatch=self._npatch,
            rng=self.rng,
        )

        self.q_tc_cat = treecorr.Catalog(
            ra=ra,
            dec=dec,
            g1=dg1,
            g2=dg2,
            w=w_tot,
            config=self.TC_config,
            patch_centers=self.p_tc_cat.patch_centers,
            rng=self.rng,
        )

        self.w_tc_cat = treecorr.Catalog(
            ra=ra,
            dec=dec,
            g1=w1,
            g2=w2,
            w=w_tot,
            config=self.TC_config,
            patch_centers=self.p_tc_cat.patch_centers,
            rng=self.rng,
        )

    def set_rho_catalog_xy(self):

        x = self._star_cat["x"]
        y = self._star_cat["y"]
        T_star = self._star_cat["T"]
        g1_star = self._star_cat["g1"]
        g2_star = self._star_cat["g2"]
        w_star = self._star_cat["w"]
        g1_psf = self._psf_cat["g1"]
        g2_psf = self._psf_cat["g2"]
        w_psf = self._psf_cat["w"]
        T_psf = self._psf_cat["T"]

        # shape_std_max = 5.
        # # Outlier rejection based on the size
        # R2_thresh_star = shape_std_max * np.std(T_star[w_star == 1]) + np.mean(
        #     T_star[w_star == 1])
        # R2_thresh_psf = shape_std_max * np.std(T_psf[w_psf == 1]) + np.mean(
        #     T_psf[w_psf == 1])
        # good_stars = (np.abs(T_star) < R2_thresh_star) \
        #     & (np.abs(T_psf) < R2_thresh_psf)

        x = x
        y = y
        g1 = g1_psf
        g2 = g2_psf
        dT = (T_star - T_psf) / T_star
        dg1 = g1_star - g1_psf
        dg2 = g2_star - g2_psf
        w1 = dT * g1_star
        w2 = dT * g2_star
        w = w_star

        w_tot = w_psf * w_star

        print("mean e = ", np.nanmean(g1), np.nanmean(g2))
        print("std e = ", np.nanstd(g1), np.nanstd(g2))
        print("mean T = ", np.nanmean(T_star))
        print("std T = ", np.nanstd(T_star))
        print("mean de = ", np.nanmean(dg1), np.nanmean(dg2))
        print("std de = ", np.nanstd(dg1), np.nanstd(dg2))
        print("mean dT = ", np.nanmean(T_star - T_psf))
        print("std dT = ", np.nanstd(T_star - T_psf))
        print("mean dT/T = ", np.nanmean(dT))
        print("std dT/T = ", np.nanstd(dT))

        # Substract mean
        if False:
            g1 -= np.mean(g1)
            g2 -= np.mean(g2)
            dg1 -= np.mean(dg1)
            dg2 -= np.mean(dg2)
            dT -= np.mean(dT)
            w1 -= np.mean(w1)
            w2 -= np.mean(w2)

        self.p_tc_cat = treecorr.Catalog(
            x=x,
            y=y,
            g1=g1,
            g2=g2,
            w=w_tot,
            config=self.TC_config,
            npatch=self._npatch,
            rng=self.rng,
        )

        self.q_tc_cat = treecorr.Catalog(
            x=x,
            y=y,
            g1=dg1,
            g2=dg2,
            w=w_tot,
            config=self.TC_config,
            patch_centers=self.p_tc_cat.patch_centers,
            rng=self.rng,
        )

        self.w_tc_cat = treecorr.Catalog(
            x=x,
            y=y,
            g1=w1,
            g2=w2,
            w=w_tot,
            config=self.TC_config,
            patch_centers=self.p_tc_cat.patch_centers,
            rng=self.rng,
        )

    def set_shear_catalog(self):

        ra = self._gal_cat["ra"]
        dec = self._gal_cat["dec"]
        g1 = self._gal_cat["g1"]
        g2 = self._gal_cat["g2"]
        w = self._gal_cat["w"]

        self.gamma_tc_cat = treecorr.Catalog(
            ra=ra,
            dec=dec,
            g1=g1,
            g2=g2,
            w=w,
            config=self.TC_config,
            patch_centers=self.p_tc_cat.patch_centers,
            rng=self.rng,
        )

    def get_rho_corr(self):

        print("Rho0...")
        self.rho0 = treecorr.GGCorrelation(
            self.TC_config,
            var_method=self.var_method,
        )
        self.rho0.process(self.p_tc_cat)
        print("Rho1...")
        self.rho1 = treecorr.GGCorrelation(
            self.TC_config,
            var_method=self.var_method,
        )
        self.rho1.process(self.q_tc_cat)
        print("Rho2...")
        self.rho2 = treecorr.GGCorrelation(
            self.TC_config,
            var_method=self.var_method,
        )
        self.rho2.process(self.q_tc_cat, self.p_tc_cat)
        print("Rho3...")
        self.rho3 = treecorr.GGCorrelation(
            self.TC_config,
            var_method=self.var_method,
        )
        self.rho3.process(self.w_tc_cat)
        print("Rho4...")
        self.rho4 = treecorr.GGCorrelation(
            self.TC_config,
            var_method=self.var_method,
        )
        self.rho4.process(self.q_tc_cat, self.w_tc_cat)
        print("Rho5...")
        self.rho5 = treecorr.GGCorrelation(
            self.TC_config,
            var_method=self.var_method,
        )
        self.rho5.process(self.p_tc_cat, self.w_tc_cat)

    def get_tau_corr(self):

        print("Tau0...")
        self.tau0 = treecorr.GGCorrelation(
            self.TC_config,
            var_method=self.var_method,
        )
        self.tau0.process(self.gamma_tc_cat, self.p_tc_cat)
        print("Tau2...")
        self.tau2 = treecorr.GGCorrelation(
            self.TC_config,
            var_method=self.var_method,
        )
        self.tau2.process(self.gamma_tc_cat, self.q_tc_cat)
        print("Tau5...")
        self.tau5 = treecorr.GGCorrelation(
            self.TC_config,
            var_method=self.var_method,
        )
        self.tau5.process(self.gamma_tc_cat, self.w_tc_cat)

    def get_all_corr(self):

        self.get_rho_corr()
        self.get_tau_corr()
