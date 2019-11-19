import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager

from functions import *

class PSO():
    r"""M, P, maxK, x_dim, w, c1, c2, k1, k2"""
    def __init__(self, M, P, maxK, x_dim, w, c1, c2, k1, k2):
        """"""
        self.M = M
        self.P = P
        self.maxK = maxK
        self.x_dim = x_dim
        self.N = M*P
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.k1 = k1
        self.k2 = k2

    def initSigma(self, W):
        sigma = np.ones(self.M)
        sigma = sigma*W/4.0
        return sigma.astype(np.double)

    def initX(self, region):
        x = np.random.randint(np.min(region), np.max(region), (self.N, self.x_dim)).astype(np.double)
        return x

    def initV(self, Vmax):
        # v = np.random.randint(-Vmax, Vmax, (self.N, self.x_dim)).astype(np.double)
        v = np.zeros((self.N,self.x_dim)).astype(np.double)
        return v

    def initTd(self, Vmax):
        Td = np.ones(self.x_dim).astype(np.double)*0.5
        return Td

    def initG(self):
        G = np.zeros(self.x_dim).astype(np.double)
        return G

    def update_Sigma(self, sigma, func, x, W):
        fx = func(x)
        # nfx = np.sort(fx)
        index = np.argsort(fx)
        nfx = np.take_along_axis(fx, index, axis=0)
        nx = np.take_along_axis(x, np.tile(np.expand_dims(index, axis=-1), (1,self.x_dim)), axis=0)
        FitX = np.zeros(self.M)
        for i in range(self.M):
            FitX[i] = np.mean(nfx[(i*self.P):((i+1)*self.P)])
        # print(FitX)
        FitX_max = np.max(FitX)
        FitX_min = np.min(FitX)
        # print(sigma)
        sigma = (sigma)*(np.exp((self.M*FitX-np.sum(FitX))/(FitX_max-FitX_min+1e-300)))
        Treshhold = W/4.0
        sigma %= Treshhold 
        sigma += 1e-300
        # w = np.array([0.1,0.3,0.5,0.7,0.9])
        # b = np.array([0,W/4000.0,W/400.0,W/40.0,W/8.0])
        # sigma = sigma*w + b
        # sigma %= W/4.0
        # sigma += 1e-300
        # nx = 0

        return sigma, nx

    def update_V(self, v, x, pb, gb, i):
        w = self.w[0]- (1.0*i/(1.0*self.maxK))*(self.w[0]-self.w[1])
        nv = w*v + self.c1*np.random.rand()*(pb-x) + self.c1*np.random.rand()*(gb-x)
        return nv

    def escape(self, v, x, func, Td, G, sigma, Vmax):
        rmax = np.sign(np.random.randn())*np.random.rand(*v.shape)*Vmax
        rsig = np.random.randn(*v.shape)

        fx = func(x+rmax)
        tmpx = np.repeat(x,self.M,0)
        tmpx += np.repeat(rsig,self.M,0)*np.expand_dims(np.tile(sigma,self.N),-1)
        # tmpx += np.tile(rsig,(self.M,1))*np.expand_dims(np.tile(sigma,self.N),-1)
        tmpf = func(tmpx)
        tmpfx = np.zeros(fx.shape)
        for i in range(self.N):
            tmpfx = np.min(tmpf[(i*self.M):((i+1)*self.M)])

        mask_vtd = np.abs(v)<Td
        mask_fx = np.repeat(np.expand_dims(tmpfx<fx,-1),self.x_dim,-1)

        v[mask_vtd&mask_fx] = rsig[mask_vtd&mask_fx]*1.0
        v[mask_vtd&~mask_fx] = rmax[mask_vtd&~mask_fx]*1.0

        G += np.sum(mask_vtd.astype(np.int), axis=0)
        gmask = G>self.k1
        G[gmask] = 0.0
        Td[gmask] /= 1.0*self.k2

        return v, Td, G

        


    def find_min(self, func, region):
        W = np.abs(region[0]-region[1]).astype(np.double)
        Vmax = W/4.0
        sigma = self.initSigma(W)
        x = self.initX(region)
        v = self.initV(Vmax)
        Td = self.initTd(Vmax)
        G = self.initG()
        pbfx = func(x)
        # print(x.shape)
        # print(pbfx.shape)
        # print(np.unravel_index(pbfx.argmin(), pbfx.shape))
        pb = x*1.0
        gbfx = np.min(pbfx)*1.0
        gb = pb[pbfx.argmin()]*1.0
        plt_sigma = np.expand_dims(sigma, -1)
        plt_x = np.array([0])
        plt_gbfx = gbfx*1.0
        for i in range(self.maxK):
            print(func(gb))
            # print(gb)
            fx = func(x)
            mask = fx<pbfx
            pbfx[mask] = fx[mask]*1.0
            pb[mask] = x[mask]*1.0
            if gbfx >= np.min(pbfx):
                gbfx = np.min(pbfx)*1.0
                gb = pb[pbfx.argmin()]*1.0
            
            v = self.update_V(v, x, pb, gb, i)
            v, Td, G = self.escape(v, x, func, Td, G, sigma, Vmax)
            # v[v>Vmax] = Vmax
            # v[v<-Vmax] = -Vmax
            x += v
            x[x>np.max(region)] = np.max(region)
            x[x<np.min(region)] = np.min(region)
            sigma, _ = self.update_Sigma(sigma, func, x, W)

            plt_sigma = np.hstack([plt_sigma,np.expand_dims(sigma,-1)])
            plt_x = np.hstack([plt_x, i])
            plt_gbfx = np.hstack([plt_gbfx, gbfx])
        return gbfx, gb, plt_sigma, plt_x, plt_gbfx


if __name__ == "__main__":
    # print(Rastrigrin(np.array([0,0,0,0])))
    # fm = FontManager()
    # mat_fonts = set(f.name for f in fm.ttflist)
    # print(mat_fonts)
    
    
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    
    model = PSO(5, 20, 6000, 30, [0.95, 0.4], 1.4, 1.4, 5, 10)
    gbfx, gb, plt_sigma, plt_x, plt_gbfx = model.find_min(Quadric, [-100, 100])
    print(gbfx, gb)
    plt.figure()
    for i in range(model.M):
        plt.plot(plt_x, plt_sigma[i],label="Sigma"+str(i))
    plt.legend()
    plt.figure()
    # plt.plot([1,2,3,4,5],[1,2,3,4,5])    
    plt.plot(plt_x, np.log(plt_gbfx))
    plt.xlabel("迭代次数")
    plt.ylabel("种群适应度的对数值log(F)")
    plt.show()

