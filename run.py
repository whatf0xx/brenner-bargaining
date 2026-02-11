from dataclasses import dataclass


@dataclass
class Agent:
    p: float
    rho: float
    v: float

    def util(self, other: "Agent") -> float:
        p_i, rho_i, v_i  = self.p, self.rho, self.v
        p_j, rho_j, v_j = other.p, other.rho, other.v

        return 0.5 * (p_i * (1 + p_i / v_j) \
                    + p_j * (1 - p_j / v_i))

    def dudp(self, other: "Agent") -> float:
        p_i = self.p
        v_j = other.v
        return 0.5 * (1 + p_i / v_j)

    def dudrho(self, other: "Agent") -> float:
        p_i = self.p
        v_j = other.v
        return 0.5 * p_i / v_j

    def dudv(self, other: "Agent") -> float:
        rho_j, p_j = other.rho, other.p
        v_i = self.v
        return 0.5 * rho_j * p_j / v_i**2


if __name__ == "__main__":
    peasant = Agent(
        1.0,
        0.2,
        0.1,
    )

    elite = Agent(
        0.2,
        0.4,
        5.0
    )

    peasant_u = peasant.util(elite)
    elite_u = elite.util(peasant)
    print(f"{peasant_u=:.3f}")
    print(f"{elite_u=:.3f}")
        
