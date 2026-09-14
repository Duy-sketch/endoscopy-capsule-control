import numpy as np

from endoscopy_capsule_control.dynamics import (
    FirstOrderCurrentDynamics,
)


def main():
    physics_dt = 0.001
    tau_current = 0.020
    current_limit = 20.0

    model = FirstOrderCurrentDynamics(
        time_constant=tau_current,
        dt=physics_dt,
        current_limit=current_limit,
    )

    print(
        "alpha =",
        model.alpha,
    )

    current = 15.0
    command = 20.0

    print(
        "\nInitial current [A] =",
        current,
    )

    print(
        "Command current [A] =",
        command,
    )

    print("\nCurrent response:")

    for step in range(1, 101):
        current = model.step(
            actual_current=current,
            commanded_current=command,
        )

        if step in [
            1,
            5,
            10,
            20,
            50,
            100,
        ]:
            time_ms = (
                step
                * physics_dt
                * 1000.0
            )

            print(
                f"{time_ms:6.1f} ms : "
                f"{current:.6f} A"
            )

    # --------------------------------------------
    # Vector test for two coils
    # --------------------------------------------

    actual = np.array([
        -15.0,
        +15.0,
    ])

    command = np.array([
        -18.0,
        +17.0,
    ])

    new_actual = model.step(
        actual_current=actual,
        commanded_current=command,
    )

    print(
        "\nTwo-coil test:"
    )

    print(
        "actual before =",
        actual,
    )

    print(
        "command       =",
        command,
    )

    print(
        "actual after  =",
        new_actual,
    )


if __name__ == "__main__":
    main()