import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from dataclasses import dataclass


@dataclass
class Agent:
    p: float
    rho: float
    v: float

    def threat(self, other: "Agent") -> float:
        p_i, rho_i  = self.p, self.rho
        v_j = other.v
        return rho_i * p_i / v_j

    def nash(self, other: "Agent") -> float:
        p_i, rho_i, v_i  = self.p, self.rho, self.v
        p_j, rho_j, v_j = other.p, other.rho, other.v
        return 0.5 * (p_i * (1 + rho_i / v_j) \
                    + p_j * (1 - rho_j / v_i))
        

    def dudp(self, other: "Agent") -> float:
        rho_i = self.rho
        v_j = other.v
        return 0.5 * (1 + rho_i / v_j)

    def dudrho(self, other: "Agent") -> float:
        p_i = self.p
        v_j = other.v
        return 0.5 * p_i / v_j

    def dudv(self, other: "Agent") -> float:
        rho_j, p_j = other.rho, other.p
        v_i = self.v
        return 0.5 * rho_j * p_j / v_i**2

    def invest(self, other: "Agent", i: float):
        dudp = self.dudp(other)
        dudrho = self.dudrho(other) * 0.1
        dudv = self.dudv(other)
        tot = dudp + dudrho + dudv
        self.p += i * dudp / tot
        self.rho += i * dudrho / tot
        self.v += i * dudv / tot


def step(peasant: Agent, elite: Agent, scale: float):
    peasant_nash = peasant.nash(elite)
    elite_nash = elite.nash(peasant)
    peasant_threat = peasant.threat(elite)
    elite_threat = elite.threat(peasant)

    agree = peasant_nash > peasant_threat and elite_nash > elite_threat
    if agree:
        peasant_u = peasant_nash
        elite_u = elite_nash
    else:
        peasant_u = peasant_threat
        elite_u = elite_threat

    peasant.invest(elite, peasant_u * scale)
    elite.invest(peasant, elite_u * scale)
    return peasant_u, elite_u, agree


if __name__ == "__main__":
    peasant = Agent(
        0.9,
        0.1,
        0.6,
    )

    elite = Agent(
        0.2,
        0.05,
        1.4,
    )

    peasants = [deepcopy(peasant)]
    elites = [deepcopy(elite)]
    peasant_nash = peasant.nash(elite)
    elite_nash = elite.nash(peasant)
    peasant_threat = peasant.threat(elite)
    elite_threat = elite.threat(peasant)

    if peasant_nash > peasant_threat and elite_nash > elite_threat:
        peasant_u = peasant_nash
        elite_u = elite_nash
        agreements = [True]
    else:
        peasant_u = peasant_threat
        elite_u = elite_threat
        agreements = [False]

    p_u = [peasant_u]
    e_u = [elite_u]
    t = range(100)        
    for _ in t:
        peasant_u, elite_u, agree = step(peasant, elite, 0.02)
        agreements.append(agree)
        p_u.append(peasant_u)
        e_u.append(elite_u)
        peasants.append(Agent(peasant.p, peasant.rho, peasant.v))
        elites.append(Agent(elite.p, elite.rho, elite.v))

    plt.figure()
    plt.title("Utility")
    plt.plot(p_u, label="peasant")
    plt.plot(e_u, label="elite")
    for ti, agreement, p, e in zip(t, agreements, p_u, e_u):
        if agreement:
            plt.plot(ti, p, "bo", alpha=0.3)
            plt.plot(ti, e, "bo", alpha=0.3)
        else:
            plt.plot(ti, p, "ro", alpha=0.3)
            plt.plot(ti, e, "ro", alpha=0.3)
    plt.legend()

    plt.figure()
    plt.title("Productivity")
    plt.plot([p.p for p in peasants], label="peasant")
    plt.plot([e.p for e in elites], label="elite")
    plt.yscale("log")
    plt.legend()
        
    plt.figure()
    plt.title("Violence")
    plt.plot([p.v for p in peasants], label="peasant")
    plt.plot([e.v for e in elites], label="elite")
    plt.yscale("log")
    plt.legend()

    plt.figure()
    plt.title("Resilience")
    plt.plot([p.rho for p in peasants], label="peasant")
    plt.plot([e.rho for e in elites], label="elite")
    plt.yscale("log")
    plt.legend()

    plt.show()
