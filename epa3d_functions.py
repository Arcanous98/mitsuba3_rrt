import os
from os.path import realpath, join, dirname
import sys

import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import numpy as np


class EpanechnikovPrimitive(Primitive):
    def __init__(self, props):
        super().__init__(props)

    def integrate_tr(self,
                    ellipsoid: Ellipsoid,
                    ray: mi.Ray3f,
                    t_0: mi.Float,
                    t_1: mi.Float,
                    sigma: mi.Float,
                    active: mi.Bool) -> mi.Float:

        p = ellipsoid.rot * (ray.o - ellipsoid.mean)

        px, py, pz = p[0], p[1], p[2]
        px2, py2, pz2 = p[0] * p[0], p[1] * p[1], p[2] * p[2]
        sx, sy, sz = ellipsoid.scale[0], ellipsoid.scale[1], ellipsoid.scale[2]
        sx2, sy2, sz2 = sx * sx, sy* sy, sz * sz
        wx, wy, wz = ray.d[0], ray.d[1], ray.d[2]
        wx2, wy2, wz2 = wx * wx, wy * wy, wz * wz

        K0 = sx2*sy2*wz2 + sy2*sz2*wx2 + sx2*sz2*wy2
        K1 = ((3*px2-21*sx2)*sy2+3*py2*sx2)*sz2+3*pz2*sx2*sy2
        K2 = pz*sx2*sy2*wz+py*sx2*sz2*wy+px*sy2*sz2*wx
        K3 = sx*sy*sz 
        K_norm = 15/(8*7**(3/2)*dr.pi*K3)
        
        tau = K_norm * (((K0*t_0**3+3*K2*t_0**2+K1*t_0)-(K0*t_1**3+3*K2*t_1**2+K1*t_1))/(21*K3*K3))
        tr = dr.exp(-tau * sigma)

        return dr.select(active, tr, 1.0)
    
    def pdf(self,
            p: mi.Point3f,
            ellipsoid: Ellipsoid,
            active: mi.Bool) -> mi.Float:
        p = ellipsoid.rot * (p - ellipsoid.mean)

        px2, py2, pz2 = p[0] * p[0], p[1] * p[1], p[2] * p[2]
        sx, sy, sz = ellipsoid.scale[0], ellipsoid.scale[1], ellipsoid.scale[2]
        sx2, sy2, sz2 = sx * sx, sy* sy, sz * sz

        K3 = sx*sy*sz 
        K_norm = 15/(8*7**(3/2)*dr.pi*K3)

        density = K_norm * (1 - 0.1428571 * (px2/sx2 + py2/sy2 + pz2/sz2)) # 0.1428571 = 1/7

        return dr.select(active, density, 0.0)
