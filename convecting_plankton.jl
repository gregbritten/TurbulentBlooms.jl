# # Critical turbulence hypothesis

using Printf
using JLD2
using Plots

using Oceananigans
using Oceananigans.Utils
using Oceananigans.Grids
using Oceananigans.Advection
using Oceananigans.AbstractOperations
using Oceananigans.OutputWriters
using Oceananigans.Fields
using Oceananigans.Diagnostics: FieldMaximum

# Parameters

Nh = 64     # Horizontal resolution
Nz = 64     # Vertical resolution
Lh = 64     # Domain width
Lz = 64     # Domain height
Qh = 1000   # Surface heat flux (W m⁻²)
 ρ = 1026   # Reference density (kg m⁻³)
cᴾ = 3991   # Heat capacity (J (ᵒC)⁻¹ m⁻²)
 α = 2e-4   # Kinematic thermal expansion coefficient (ᵒC m⁻¹)
 g = 9.81   # Gravitational acceleration (m s⁻²)
N∞ = 9.5e-3 # s⁻²
 f = 1e-4   # s⁻¹

buoyancy_flux_parameters = (initial_buoyancy_flux = α * g * Qh / (ρ * cᴾ), # m³ s⁻²
                            start_ramp_down = 12hours,
                            shut_off = 1day)

planktonic_parameters = (sunlight_attenuation_scale = 16.0,
                         surface_growth_rate = 1/day,
                         mortality_rate = 0.1/day)

P₀ = 1
initial_plankton_concentration(x, y, z) = P₀ # μM

initial_time_step = 10
max_time_step = 2minutes
stop_time = 1hours
output_interval = hour / 2

@info """ *** Parameters ***

    Resolution:                        ($Nh, $Nh, $Nz)
    Domain:                            ($Lh, $Lh, $Lz) m
    Initial heat flux:                 $(Qh) W m⁻²
    Initial buoyancy flux:             $(@sprintf("%.2e", buoyancy_flux_parameters.initial_buoyancy_flux)) m² s⁻³
    Cooling starts ramping down:       $(prettytime(buoyancy_flux_parameters.start_ramp_down))
    Cooling shuts off:                 $(prettytime(buoyancy_flux_parameters.shut_off))
    Plankton surface growth rate:      $(day * planktonic_parameters.surface_growth_rate) day⁻¹
    Plankton mortality rate:           $(day * planktonic_parameters.mortality_rate) day⁻¹
    Sunlight attenuation length scale: $(planktonic_parameters.sunlight_attenuation_scale) m

"""


# Grid

grid = RegularCartesianGrid(size=(Nh, Nh, Nz), extent=(Lh, Lh, Lz))

# Boundary conditions

# Buoyancy flux
delayed_ramp_down(t, start, shutoff) =
    ifelse(t < start, 1.0,
    ifelse(t < shutoff, (shutoff - t) / (shutoff - start), 0.0))

buoyancy_flux(x, y, t, θ) = θ.initial_buoyancy_flux * delayed_ramp_down(t, θ.start_ramp_down, θ.shut_off)

buoyancy_top_bc = BoundaryCondition(Flux, buoyancy_flux, parameters=buoyancy_flux_parameters)
buoyancy_bot_bc = BoundaryCondition(Gradient, N∞^2)
                                                   
buoyancy_bcs = TracerBoundaryConditions(grid, top = buoyancy_top_bc, bottom = buoyancy_bot_bc)

# Plankton dynamics

growing_and_grazing(z, P, h, μ₀, m) = (μ₀ * exp(z / h) - m) * P

plankton_forcing_func(x, y, z, t, P, θ) = growing_and_grazing(z, P,
                                                              θ.sunlight_attenuation_scale,
                                                              θ.surface_growth_rate,
                                                              θ.mortality_rate)

plankton_forcing = Forcing(plankton_forcing_func, field_dependencies=:plankton,
                           parameters=planktonic_parameters)

# Model setup

model = IncompressibleModel(
                   grid = grid,
              advection = UpwindBiasedFifthOrder(),
            timestepper = :RungeKutta3,
                closure = AnisotropicMinimumDissipation(),
               coriolis = FPlane(f=f),
                tracers = (:b, :plankton),
               buoyancy = BuoyancyTracer(),
                forcing = (plankton=plankton_forcing,),
    boundary_conditions = (b=buoyancy_bcs,)
)

# Initial condition

Ξ(z) = randn() * z / grid.Lz * (1 + z / grid.Lz) # noise

initial_buoyancy(x, y, z) = N∞^2 * z + N∞^2 * grid.Lz * 1e-6 * Ξ(z)

set!(model, b=initial_buoyancy, plankton=initial_plankton_concentration)

# Simulation setup

wizard = TimeStepWizard(cfl=1.0, Δt=Float64(initial_time_step), max_change=1.1, max_Δt=Float64(max_time_step))

wmax = FieldMaximum(abs, model.velocities.w)
Pmax = FieldMaximum(abs, model.tracers.plankton)

start_time = time_ns() # so we can print the total elapsed wall time

progress_message(sim) = @info @sprintf(
    "i: % 4d, t: % 12s, Δt: % 12s, max(|w|) = %.1e ms⁻¹, max(|P|) = %.1e μM, wall time: %s\n",
    sim.model.clock.iteration, prettytime(model.clock.time),
    prettytime(wizard.Δt), wmax(sim.model), Pmax(sim.model),
    prettytime((time_ns() - start_time) * 1e-9))

simulation = Simulation(model, Δt=wizard, stop_time=stop_time,
                        iteration_interval=10, progress=progress_message)

u, v, w = model.velocities
P = model.tracers.plankton

_P_  = AveragedField(P, dims=(1, 2))
_wP_ = AveragedField(w * P, dims=(1, 2))
_Pz_ = AveragedField(∂z(P), dims=(1, 2))

simulation.output_writers[:fields] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                     schedule = TimeInterval(output_interval),
                     prefix = "convecting_plankton_fields",
                     force = true)

simulation.output_writers[:averages] =
    JLD2OutputWriter(model, (P = _P_, wP = _wP_, Pz = _Pz_),
                     schedule = TimeInterval(output_interval),
                     prefix = "convecting_plankton_averages",
                     force = true)

run!(simulation)

# Movie

file = jldopen(simulation.output_writers[:fields].filepath)
averages_file = jldopen(simulation.output_writers[:averages].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

xw, yw, zw = nodes(model.velocities.w)
xp, yp, zp = nodes(model.tracers.plankton)

@info "Making a movie about plankton..."

anim = @animate for (i, iteration) in enumerate(iterations)

    @info "Plotting frame $i from iteration $iteration..."
    
    t = file["timeseries/t/$iteration"]
    w = file["timeseries/w/$iteration"][:, 1, :]
    p = file["timeseries/plankton/$iteration"][:, 1, :]

    P = averages_file["timeseries/P/$iteration"][1, 1, :]
    wP = averages_file["timeseries/wP/$iteration"][1, 1, :]
    Pz = averages_file["timeseries/Pz/$iteration"][1, 1, :]

    κᵉᶠᶠ = @. - wP / Pz

    # Normalize profiles
    P ./= P₀
    wP ./= maximum(abs, wP)
    Pz ./= maximum(abs, Pz)

    w_max = maximum(abs, w) + 1e-9
    w_lim = 0.8 * w_max

    p_min = minimum(p) - 1e-9
    p_max = maximum(p) + 1e-9
    p_lim = 2

    w_levels = vcat([-w_max], range(-w_lim, stop=w_lim, length=21), [w_max])
    p_levels = collect(range(p_min, stop=p_lim, length=20))
    p_max > p_lim && push!(p_levels, p_max)

    kwargs = (xlabel="x", ylabel="y", aspectratio=1, linewidth=0, colorbar=true,
              xlims=(0, model.grid.Lx), ylims=(-model.grid.Lz, 0))

    w_plot = contourf(xw, zw, w';
                       color = :balance,
                      levels = w_levels,
                       clims = (-w_lim, w_lim),
                      kwargs...)

    p_plot = contourf(xp, zp, p';
                       color = :matter,
                      levels = p_levels,
                       clims = (p_min, p_lim),
                      kwargs...)

    profile_plot = plot(P, zp, label = "⟨P⟩ / P₀", linewidth = 2,
                        xlabel = "Normalized plankton statistics",
                        legend = :bottom,
                        ylabel = "z (m)")

    plot!(profile_plot, wP, zw, label = "⟨wP⟩ / max|wP|", linewidth = 2)
    plot!(profile_plot, Pz, zw, label = "⟨∂_z P⟩ / max|∂_z P|", linewidth = 2)

    κᵀ_plot = plot(κᵉᶠᶠ, zw, linewidth = 2, label = nothing, xlims = (0.8, 2),
                   ylabel = "z (m)", xlabel = "turbulent diffusivity (m² s⁻¹)")

    w_title = @sprintf("w(y=0, t=%s) (m s⁻¹)", prettytime(t))
    p_title = @sprintf("P(y=0, t=%s) (μM)", prettytime(t))

    plot(w_plot, p_plot, profile_plot, κᵀ_plot,
         title=[w_title p_title "Plankton statistics" "Turbulent diffusivity"],
         link = :y,
         layout=Plots.grid(1, 4, widths=[0.4, 0.4, 0.1, 0.1]), size=(1700, 400))
end

gif(anim, "convecting_plankton.gif", fps = 8) # hide
