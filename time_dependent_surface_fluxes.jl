# # Windy convection with surface waves

# This script runs a simulation of ocean turbulence driven by both convection
# and wind in the presence of surface waves.

using Oceananigans, Random, Printf, Statistics

# # Set up the model
#
# ## Grid

using Oceananigans.Grids

grid = RegularCartesianGrid(
                            size = (64, 64, 64),
                               x = (0, 128),
                               y = (0, 128),
                               z = (-64, 0)
                            )
# ## Boundary conditions

using Oceananigans.BoundaryConditions

const day = Oceananigans.Utils.day
const f = 1e-4
const τ = 1e-4
const Qᵇ₀ = 1e-8

Qᵇ(x, y, t) = Qᵇ₀ * (1 + cos(2π/day * t))
Qᵘ(x, y, t) =   τ * cos(2π * f * t)
Qᵛ(x, y, t) = - τ * sin(2π * f * t)

N² = 1e-5

b_bcs = TracerBoundaryConditions(grid, top = TracerBoundaryCondition(Flux, :z, Qᵇ),
                                       bottom = BoundaryCondition(Gradient, N²))

u_bcs = UVelocityBoundaryConditions(grid, top = UVelocityBoundaryCondition(Flux, :z, Qᵘ))
v_bcs = VVelocityBoundaryConditions(grid, top = VVelocityBoundaryCondition(Flux, :z, Qᵛ))

# ## Plankton forcing

using Oceananigans.Utils: hour, minute

const p₀ = 1

@inline sunlight(z) = exp(z / 24)
const growth_rate = 1 / day
const mortality_rate = 1 / day

@inline function growth_and_death(i, j, k, grid, clock, state) 
    p = @inbounds state.tracers.plankton[i, j, k]
    z = @inbounds grid.zC[k]

    growth = growth_rate * sunlight(z) * p * (1 - p / p₀)

    death = mortality_rate * p

    return growth - death
end

# ## Stokes drift

struct SteadyStokesShear{T} <: Function
    a :: T
    k :: T
    g :: T

    function SteadyStokesShear(a, k, g=9.81; T=Float64)
        return new{T}(a, k, g)
    end
end

@inline (uˢ::SteadyStokesShear)(z, t) = 2 * (uˢ.a * uˢ.k)^2 * sqrt(uˢ.g * uˢ.k) * exp(2 * uˢ.k * z)

# ## Model instantiation

using Oceananigans
using Oceananigans.SurfaceWaves: UniformStokesDrift

model = IncompressibleModel(architecture = CPU(),
                                    grid = grid,
                                 tracers = (:b, :plankton),
                                buoyancy = BuoyancyTracer(),
                                coriolis = FPlane(f=f),
                                 closure = AnisotropicMinimumDissipation(),
                           surface_waves = UniformStokesDrift(∂z_uˢ=SteadyStokesShear(0.8, 2π/60)),
                     boundary_conditions = (u=u_bcs, v=v_bcs, b=b_bcs),
                                 forcing = ModelForcing(plankton=growth_and_death))

# ## Initial condition

set!(model,
     b = (x, y, z) -> N² * z + 1e-6 * N² * grid.Lz * exp(z / 8) * randn())

# # Prepare the simulation

using Oceananigans.Utils: hour, minute
using LESbrary.Utils: SimulationProgressMessenger

# Adaptive time-stepping
wizard = TimeStepWizard(cfl=0.2, Δt=1e-1, max_change=1.1, max_Δt=10.0)

simulation = Simulation(model, Δt=wizard, stop_time=12hour, progress_frequency=100,
                        progress=SimulationProgressMessenger(model, wizard))

# ## Checkpointer

using Oceananigans.Utils: GiB
using Oceananigans.OutputWriters: Checkpointer, JLD2OutputWriter, FieldOutputs

prefix = @sprintf("windy_convection_with_plankton, Qu%.1e_Qb%.1e_Nsq%.1e_N%d",
                  abs(τ), Qᵇ₀, N², grid.Nz)

data_directory = joinpath(@__DIR__, "..", "data", prefix) # save data in /data/prefix

simulation.output_writers[:checkpointer] = Checkpointer(model, force = true,
                                                            interval = 6hour, # every quarter period
                                                                 dir = data_directory,
                                                              prefix = prefix * "_checkpoint")

simulation.output_writers[:fields] = JLD2OutputWriter(model, FieldOutputs(merge(model.velocities, model.tracers)),
                                                              force = true,
                                                           interval = 1hour,
                                                                dir = data_directory,
                                                             prefix = prefix * "_fields")

# # Run
run!(simulation)

exit() # Release GPU memory
