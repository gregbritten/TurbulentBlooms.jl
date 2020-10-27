# Reproduces Taylor and Ferrari (2011)

using Random
using Printf
using Plots
using JLD2

using Oceananigans
using Oceananigans.Utils
using Oceananigans.Buoyancy
using Oceananigans.OutputWriters

using Oceananigans.Grids
using Oceananigans.Advection
using Oceananigans.Diagnostics: FieldMaximum

Nh = 64
Nz = 64
Lh = 256
Lz = 128
Qʰ = 100  # W m⁻², surface _heat_ flux
ρₒ = 1026 # kg m⁻³, average density at the surface of the world ocean
cᴾ = 3991 # J K⁻¹ s⁻¹, typical heat capacity for seawater
N∞ = 9.5e-3 # s⁻¹, initial buoyancy frequency

surface_temperature_flux_parameters = (
                                       transition_time = 1day,
                                       shutoff_time = 3day
                                      )

plankton_forcing_parameters = (
                               surface_growth_rate = 1/day,
                               sunlight_attenuation_length = 5.0,
                               mortality_rate = 0.1/day,
                              )
                           
# Setup

Qᵀ = Qʰ / (ρₒ * cᴾ) # K m⁻¹ s⁻¹, surface _temperature_ flux

grid = RegularCartesianGrid(size=(Nh, Nh, Nz), x=(0, Lh), y=(0, Lh), z=(-Lz, 0))

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4), constant_salinity=true)

α = buoyancy.equation_of_state.α
g = buoyancy.gravitational_acceleration

dTdz = α * g * N∞^2 # ᵒC m⁻¹

#
# ∂⟨T⟩/∂t + ∇ ⋅ F = 0
#
# ∂⟨T⟩/∂t = - ∇ ⋅ F
#

ramp(t, t₁, t₂) = (t₂ - t) / (t₂ - t₁)

ramping_temperature_flux(x, y, t, p) =
    ifelse(t < p.transition_time, Qᵀ,
    ifelse(t < p.shutoff_time,    Qᵀ * ramp(t, p.transition_time, p.shutoff_time),
                                  0.0)
    )


surface_temperature_flux =
    BoundaryCondition(Flux,
                      ramping_temperature_flux,
                      parameters = surface_temperature_flux_parameters)

T_bcs = TracerBoundaryConditions(grid, 
                                 top = surface_temperature_flux,
                                 bottom = BoundaryCondition(Gradient, dTdz))

#
# Forcing: ∂p/∂t = (μ(z) - m) * p
#
# θ: parameters (for a statistician :-D)
# 

@inline growth_and_grazing(x, y, t, z, P, θ) =
    (θ.surface_growth_rate * exp(z / θ.sunlight_attenuation_length) - θ.mortality_rate) * P

plankton_forcing = Forcing(growth_and_grazing, field_dependencies=:P,
                           parameters=plankton_forcing_parameters)

model = IncompressibleModel(architecture = CPU(),
                               advection = UpwindBiasedFifthOrder(),
                             timestepper = :RungeKutta3,
                                    grid = grid,
                                 tracers = (:T, :P),
                                coriolis = FPlane(f=1e-4),
                                buoyancy = buoyancy,
                                 closure = AnisotropicMinimumDissipation(),
                     boundary_conditions = (T=T_bcs,),
                                 forcing = (P=plankton_forcing,))

# Initial condition

# Random noise concentrated in the top 8 meters
Ξ(z) = randn() * exp(z / 8)

Tᵢ(x, y, z) = 20 + dTdz * z + dTdz * model.grid.Lz * 1e-6 * Ξ(z)

set!(model, T=Tᵢ, P=1)

wizard = TimeStepWizard(cfl=1.0, Δt=1.0, max_change=1.1, max_Δt=1.0)

# Nice progress messaging is helpful:

wmax = FieldMaximum(abs, model.velocities.w)

start_time = time_ns() # so we can print the total elapsed wall time

## Print a progress message
progress_message(sim) =
    @printf("i: %04d, t: %s, Δt: %s, wmax = %.1e ms⁻¹, wall time: %s\n",
            sim.model.clock.iteration, prettytime(model.clock.time),
            prettytime(wizard.Δt), wmax(sim.model),
            prettytime((time_ns() - start_time) * 1e-9))

# We then set up the simulation:

simulation = Simulation(model,
                        Δt = wizard,
                        stop_time = 6day,
                        iteration_interval = 10,
                        progress = progress_message)

## Create a NamedTuple with eddy viscosity
eddy_viscosity = (νₑ = model.diffusivities.νₑ,)

simulation.output_writers[:slices] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, eddy_viscosity),
                           prefix = "ocean_wind_mixing_and_convection",
                     field_slicer = FieldSlicer(j=Int(grid.Ny/2)),
                         schedule = TimeInterval(1minute),
                            force = true)

run!(simulation)

# Turbulence visualization

## Coordinate arrays
xw, yw, zw = nodes(model.velocities.w)
xT, yT, zT = nodes(model.tracers.T)

## Open the file with our data
file = jldopen(simulation.output_writers[:slices].filepath)

## Extract a vector of iterations
iterations = parse.(Int, keys(file["timeseries/t"]))

""" Returns colorbar levels equispaced between `(-clim, clim)` and encompassing the extrema of `c`. """
function divergent_levels(c, clim, nlevels=21)
    levels = range(-clim, stop=clim, length=nlevels)
    cmax = maximum(abs, c)
    return ((-clim, clim), clim > cmax ? levels : levels = vcat([-cmax], levels, [cmax]))
end

""" Returns colorbar levels equispaced between `clims` and encompassing the extrema of `c`."""
function sequential_levels(c, clims, nlevels=20)
    levels = range(clims[1], stop=clims[2], length=nlevels)
    cmin, cmax = minimum(c), maximum(c)
    cmin < clims[1] && (levels = vcat([cmin], levels))
    cmax > clims[2] && (levels = vcat(levels, [cmax]))
    return clims, levels
end

# We start the animation at `t = 10minutes` since things are pretty boring till then:

times = [file["timeseries/t/$iter"] for iter in iterations]

anim = @animate for (i, iter) in enumerate(iterations)

    @info "Drawing frame $i from iteration $iter..."

    t = file["timeseries/t/$iter"]
    w = file["timeseries/w/$iter"][:, 1, :]
    T = file["timeseries/T/$iter"][:, 1, :]
    P = file["timeseries/P/$iter"][:, 1, :]

    wmax = maximum(abs, w)
    Tmax = maximum(abs, T)
    Pmax = maximum(abs, P)

    wlims, wlevels = divergent_levels(w, 0.8 * wmax)
    Tlims, Tlevels = sequential_levels(T, (19.7, 19.99))
    Plims, Plevels = sequential_levels(P, (0.0, 0.8 * Pmax))

    kwargs = (linewidth=0, xlabel="x (m)", ylabel="z (m)", aspectratio=1,
              xlims=(0, grid.Lx), ylims=(-grid.Lz, 0))

    w_plot = contourf(xw, zw, w'; color=:balance, clims=wlims, levels=wlevels, kwargs...)
    T_plot = contourf(xT, zT, T'; color=:thermal, clims=Tlims, levels=Tlevels, kwargs...)
    P_plot = contourf(xT, zT, P'; color=:haline,  clims=Slims, levels=Slevels, kwargs...)

    w_title = @sprintf("vertical velocity (m s⁻¹), t = %s", prettytime(t))
    T_title = "temperature (ᵒC)"
    S_title = "salinity (g kg⁻¹)"
    P_title = "plankton (?)"
                       
    ## Arrange the plots side-by-side.
    plot(w_plot, T_plot, P_plot, layout=(1, 3), size=(1200, 600),
         title=[w_title T_title P_title])

    iter == iterations[end] && close(file)
end

gif(anim, "critical_turbulence_hypothesis.gif", fps = 8) # hide
