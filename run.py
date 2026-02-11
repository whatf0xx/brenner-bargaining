import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from dataclasses import dataclass


@dataclass
class Agent:
    p: float
    rho: float
    v: float

    def util(self, other: "Agent") -> float:
        p_i, rho_i, v_i  = self.p, self.rho, self.v
        p_j, rho_j, v_j = other.p, other.rho, other.v
        threat = rho_i * p_i / v_j
        nash = 0.5 * (p_i * (1 + rho_i / v_j) \
                    + p_j * (1 - rho_j / v_i))

        return max(threat, nash)

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
        options = {
            "p": self.dudp(other),
            "rho": self.dudrho(other),
            "v": self.dudv(other),
        }
        pick = max(options, key=lambda k: options[k])
        new_val = getattr(self, pick) + i
        setattr(self, pick, new_val)


def step(peasant: Agent, elite: Agent, scale: float):
    peasant_u = peasant.util(elite)
    elite_u = elite.util(peasant)
    peasant.invest(elite, peasant_u * scale)
    elite.invest(peasant, elite_u * scale)


if __name__ == "__main__":
    peasant = Agent(
        0.6,
        0.2,
        0.1,
    )

    elite = Agent(
        0.3,
        0.2,
        0.2,
    )

    peasants = [deepcopy(peasant)]
    elites = [deepcopy(elite)]
    p_u = [peasant.util(elite)]
    e_u = [elite.util(peasant)]
    t = range(100)        
    for _ in t:
        p_u.append(peasant.util(elite))
        e_u.append(elite.util(peasant))
        step(peasant, elite, 0.01)
        peasants.append(Agent(peasant.p, peasant.rho, peasant.v))
        elites.append(Agent(elite.p, elite.rho, elite.v))

    plt.figure()
    plt.title("Utility")
    plt.plot(p_u, label="peasant")
    plt.plot(e_u, label="elite")
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
