# API Documentation

## Create a model
```@docs
@ReactionNetwork
```

## Modify a model

We list common transition attributes. When being specified using the `@ReactionNetwork` macro they can be conveniently referred to using their shorthand description.

| attribute | shorthand | interpretation |
| :----- | :----- | :----- |
| `transPriority` | `priority` | priority of a transition (influences resource allocation) |
| `transProbOfSuccess` | `probability` `prob` `pos` | probability that a transition terminates successfully |
| `transCapacity` | `cap` `capacity` | maximum number of concurrent instances of the transition |
| `transCycleTime` | `ct` `cycletime` | duration of a transition's instance (adjusted by resource allocation) |
| `transMaxLifeTime` | `lifetime` `maxlifetime` `maxtime` `timetolive` | maximal duration of a transition's instance |
| `transPostAction` | `postAction` `post` | action to be executed once a transition's instance terminates |
| `transName` | `name` `interpretation` | name of a transition, either a string or unquoted text |

We list common species attributes:

| attribute | shorthand | interpretation |
| :----- | :----- | :----- |
| `specInitUncertainty` | `uncertainty` `stoch` `stochasticity` | uncertainty about variable's initial state (modelled as Gaussian standard deviation) |
| `specInitVal` | | initial value of a variable |

Moreover, it is possible to specify the semantics of the "rate" term. By default, at each time step `n ~ Poisson(rate * dt)` instances of a given transition will be spawned. If you want to specify the rate in terms of a cycle time, you may want to use `@ct(cycle_time)`, e.g., `@ct(ex), A --> B, ...`. This is a shorthand for `1/ex, A --> B, ...`.

For deterministic "rates", use `@deterministic(ex)`. Here, `ex` evaluates to a deterministic number (ceiled to the nearest integer) of a transition's instances to spawn per a single integrator's step. However, note that in this case, the number doesn't scale with the step length! Moreover

```@docs
@add_species
@aka
@mode
@name_transition
```

## Resource costs
```@docs
@cost
@valuation
@reward
```

## Add reactions
```@docs
@push
@jump
@periodic
```

## Set initial values, uncertainty, and solver arguments
```@docs
@prob_init
@prob_uncertainty
@prob_params
@prob_meta
```

## Model unions
```@docs
@join
@equalize
```

## Model import and export
```@docs
@import_network
@export_network
```

## Solution import and export
```@docs
@import_solution
@export_solution_as_table
@export_solution_as_csv
@export_solution
```

## Problematize,sSolve, and plot
```@docs
@problematize
@solve
@plot
```

## Optimization and fitting
```@docs
@optimize
@fit
@fit_and_plot
@build_solver
```