from vspace import *


class Solver(SolverBase):

    def __init__(self, plot=False):
        super().__init__(POS_HS, 0, z_hs=4, Lwh=6 + 6j)
        self.refresh()
        self.res2excel(1)

        if plot:
            args = 12, 0
            self.plot_hs(*args), plt.show()
            for i, k in enumerate(['', 'cos', 'sb', 'trunc']):
                self.plot_eta(k, *args, fig=plt.subplot(2, 2, i + 1))
            plt.show()


if __name__ == '__main__':
    Solver(True)
