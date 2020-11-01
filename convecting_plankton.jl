#####
##### The critical turbulence hypothesis
#####
##### Reproduces Taylor and Ferrari (2011):
#####
##### https://aslopubs.onlinelibrary.wiley.com/doi/pdf/10.4319/lo.2011.56.6.2293
#####
##### Adapted from the Oceananigans' example convecting_plankton.jl.
#####

using Printf
using JLD2

## https://discourse.julialang.org/t/unable-to-display-plot-using-the-repl-gks-errors/12826/18
ENV["GKSwstype"] = "nul"
using Plots
using Measures: pt

using Oceananigans
using Oceananigans.Utils
using Oceananigans.Grids
using Oceananigans.Advection
using Oceananigans.AbstractOperations
using Oceananigans.OutputWriters
using Oceananigans.Fields
using Oceananigans.Diagnostics: FieldMaximum

#####
##### Parameters
#####

just_remake_animation = false

Nh = 192      # Horizontal resolution
Nz = 192      # Vertical resolution
Lh = 192      # Domain width
Lz = 96       # Domain height
Qh = 10       # Surface heat flux (W m⁻²)
 ρ = 1000     # Reference density used by Taylor and Ferrari (2011) (kg m⁻³)
cᴾ = 4000     # Heat capacity used in Taylor and Ferrari (2011) (J (ᵒC)⁻¹ m⁻²)
 α = 1.64e-4  # Kinematic thermal expansion coefficient (ᵒC m⁻¹)
 g = 9.81     # Gravitational acceleration (m s⁻²)
N∞ = 9.5e-3   # Initial buoyancy frequency below the mixed layer (s⁻²)
 f = 1e-4     # Coriolis parameter (s⁻¹)

simulation_stop_time = 6day
initial_mixed_layer_depth = 50 # m

buoyancy_flux_parameters = (maximum_buoyancy_flux = α * g * Qh / (ρ * cᴾ), # m³ s⁻²
                                     stop_ramp_up = 1day,
                                  start_ramp_down = 2day,
                                         shut_off = 4day)

plankton_dynamics_parameters = (sunlight_attenuation_length = 5.0, # m
                                       surface_growth_rate = 1/day,
                                            mortality_rate = 0.1/day)

# Initial uniform plankton concentration
P₀ = 1 # μM
initial_plankton_concentration(x, y, z) = P₀ # μM

# Numerics and output parameters
initial_time_step = 30 # s
    max_time_step = 2minutes
  output_interval = hour / 2

@info """ Parameters for demonstrating the critical turbulence hypothesis

    Resolution:                        ($Nh, $Nh, $Nz)
    Domain:                            ($Lh, $Lh, $Lz) m
    Initial heat flux:                 $(Qh) W m⁻²
    Maximum buoyancy flux:             $(@sprintf("%.2e", buoyancy_flux_parameters.maximum_buoyancy_flux)) m² s⁻³
    Initial mixed layer depth:         $(initial_mixed_layer_depth) m
    Cooling increasesmping down:       $(prettytime(buoyancy_flux_parameters.start_ramp_down))
    Cooling starts ramping down:       $(prettytime(buoyancy_flux_parameters.start_ramp_down))
    Cooling shuts off:                 $(prettytime(buoyancy_flux_parameters.shut_off))
    Simulation stop time:              $(prettytime(simulation_stop_time))
    Plankton surface growth rate:      $(day * plankton_dynamics_parameters.surface_growth_rate) day⁻¹
    Plankton mortality rate:           $(day * plankton_dynamics_parameters.mortality_rate) day⁻¹
    Sunlight attenuation length scale: $(plankton_dynamics_parameters.sunlight_attenuation_length) m

"""

@info """ Notes about physical parameters:

    * Taylor and Ferrari (2011) run four simulations with Qh = {1, 10, 100, 1000} W m⁻².
    
    * Typical reference densities for ocean seawater are somewhat higher than 1000 kg m⁻³.
      The average value at the surface of the world ocean is 1026 kg m⁻³ (Roquet et al 2015, Ocean Modelling).
    
    * For heat fluxes calculated for conservative temperature use cᴾ = 3991 J (ᵒC)⁻¹ m⁻².
    
    * Our kinematic thermal expansion coefficient, α, differs from the "ordinary" thermal expansion
      coefficient. Here, α = α̃ / ρ, where α̃ is the thermal expansion coefficient and 
      ρ is the reference density.
      See https://en.wikipedia.org/wiki/Thermal_expansion#Coefficient_of_thermal_expansion.
    
    * (Is a sunlight attenuation length of 5 m reasonable?)

"""

#####
##### Grid and boundary conditions
#####

grid = RegularCartesianGrid(size=(Nh, Nh, Nz), extent=(Lh, Lh, Lz))

ramp_up_ramp_down(t, plateau, ramp_down, shutoff) =
    ifelse(t < plateau, t / plateau,
    ifelse(t < ramp_down, 1.0,
    ifelse(t < shutoff, (shutoff - t) / (shutoff - ramp_down), 0.0)))

buoyancy_flux(x, y, t, θ) = θ.maximum_buoyancy_flux * ramp_up_ramp_down(t, θ.stop_ramp_up, θ.start_ramp_down, θ.shut_off)

buoyancy_top_bc = BoundaryCondition(Flux, buoyancy_flux, parameters=buoyancy_flux_parameters)
buoyancy_bot_bc = BoundaryCondition(Gradient, N∞^2)
                                                   
buoyancy_bcs = TracerBoundaryConditions(grid, top = buoyancy_top_bc, bottom = buoyancy_bot_bc)

#####
##### Plankton dynamics
#####

growing_and_grazing(z, P, h, μ₀, m) = (μ₀ * exp(z / h) - m) * P

plankton_forcing_func(x, y, z, t, P, θ) = growing_and_grazing(z, P,
                                                              θ.sunlight_attenuation_length,
                                                              θ.surface_growth_rate,
                                                              θ.mortality_rate)

plankton_forcing = Forcing(plankton_forcing_func, field_dependencies=:plankton,
                           parameters=plankton_dynamics_parameters)

#####
##### Bottom sponge layer for u, v, w, and b
#####

gaussian_mask = GaussianMask{:z}(center=-grid.Lz, width=grid.Lz/10)

u_sponge = v_sponge = w_sponge = Relaxation(rate=2/hour, mask=gaussian_mask)

b_sponge = Relaxation(rate = 4/hour,
                      target = LinearTarget{:z}(intercept=0, gradient=N∞^2),
                      mask = gaussian_mask)

#####
##### Simulate some phytoplankton
#####

if !just_remake_animation # actually run the simulation
    
    model = IncompressibleModel(
               architecture = GPU(),
                       grid = grid,
                  advection = UpwindBiasedFifthOrder(),
                timestepper = :RungeKutta3,
                    closure = AnisotropicMinimumDissipation(),
                   coriolis = FPlane(f=f),
                    tracers = (:b, :plankton),
                   buoyancy = BuoyancyTracer(),
                    forcing = (u=u_sponge, v=v_sponge, w=w_sponge, b=b_sponge, plankton=plankton_forcing),
        boundary_conditions = (b=buoyancy_bcs,)
    )

    # Initial condition

    Ξ(z) = N∞^2 * grid.Lz * 1e-4 * randn() * exp(z / 4) # surface-concentrated noise

    stratification(x, y, z) = N∞^2 * z

    initial_buoyancy(x, y, z) =
        Ξ(z) + ifelse(z < -initial_mixed_layer_depth,
                      stratification(x, y, z),
                      stratification(x, y, -initial_mixed_layer_depth))

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

    simulation = Simulation(model, Δt=wizard, stop_time=simulation_stop_time,
                            iteration_interval=10, progress=progress_message)

    u, v, w = model.velocities
    P = model.tracers.plankton

     P̂   = AveragedField(P, dims=(1, 2, 3))
    _P_  = AveragedField(P, dims=(1, 2))
    _wP_ = AveragedField(w * P, dims=(1, 2))
    _Pz_ = AveragedField(∂z(P), dims=(1, 2))

    simulation.output_writers[:fields] =
        JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                         schedule = TimeInterval(output_interval),
                         prefix = "convecting_plankton_fields",
                         force = true)

    simulation.output_writers[:averages] =
        JLD2OutputWriter(model, (P = _P_, wP = _wP_, Pz = _Pz_, volume_averaged_P = P̂),
                         schedule = TimeInterval(output_interval),
                         prefix = "convecting_plankton_averages",
                         force = true)

    run!(simulation)
end

#####
##### Movie
#####

fields_file = jldopen("convecting_plankton_fields.jld2")
averages_file = jldopen("convecting_plankton_averages.jld2")

iterations = parse.(Int, keys(fields_file["timeseries/t"]))
times = [fields_file["timeseries/t/$iter"] for iter in iterations]
buoyancy_flux_time_series = [buoyancy_flux(0, 0, t, buoyancy_flux_parameters) for t in times] 

xw, yw, zw = nodes((Cell, Cell, Face), grid)
xp, yp, zp = nodes((Cell, Cell, Cell), grid)

function divergent_levels(c, clim, nlevels=31)
    levels = range(-clim, stop=clim, length=nlevels)
    cmax = maximum(abs, c)
    clim < cmax && (levels = vcat([-cmax], levels, [cmax]))
    return (-clim, clim), levels
end

function sequential_levels(c, clims, nlevels=31)
    levels = collect(range(clims[1], stop=clims[2], length=nlevels))
    cmin = minimum(c)
    cmax = maximum(c)
    cmin < clims[1] && pushfirst!(levels, cmin)
    cmax > clims[2] && push!(levels, cmax)
    return clims, levels
end

@info "Making a movie about plankton..."

try
    anim = @animate for (i, iteration) in enumerate(iterations)

        local w
        local p

        @info "Plotting frame $i from iteration $iteration..."
        
        w = fields_file["timeseries/w/$iteration"][:, 1, :]
        p = fields_file["timeseries/plankton/$iteration"][:, 1, :]

        P = averages_file["timeseries/P/$iteration"][1, 1, :]
        wP = averages_file["timeseries/wP/$iteration"][1, 1, :]
        Pz = averages_file["timeseries/Pz/$iteration"][1, 1, :]

        κᵉᶠᶠ = @. - wP / Pz

        # Normalize profiles
        @. P = P / P₀ - 1
        wP ./= maximum(abs, wP)
        Pz ./= maximum(abs, Pz)

        w_lim = 1e-2 # 0.8 * maximum(abs, w) + 1e-9
        p_lim = 2

        w_lims, w_levels = divergent_levels(w, w_lim)
        p_lims, p_levels = sequential_levels(p, (0.9, p_lim))

        kwargs = (xlabel="x (m)", ylabel="y (m)", aspectratio=1, linewidth=0, colorbar=true,
                  xlims=(0, grid.Lx), ylims=(-grid.Lz, 0))

        w_contours = contourf(xw, zw, w';
                               color = :balance,
                              margin = 10pt,
                              levels = w_levels,
                               clims = w_lims,
                              kwargs...)

        p_contours = contourf(xp, zp, p';
                               color = :matter,
                              margin = 10pt,
                              levels = p_levels,
                               clims = p_lims,
                              kwargs...)

        profile_plot = plot(P, zp, label = "⟨P⟩ / P₀ - 1",
                               linewidth = 4,
                                   color = :black,
                                   alpha = 0.4,
                                  margin = 20pt,
                                  xlabel = "Normalized plankton statistics",
                                  legend = :bottom,
                                  ylabel = "z (m)")

        plot!(profile_plot, wP, zw, label = "⟨wP⟩ / max|wP|", color = :steelblue, linewidth = 1.2, alpha = 0.8)
        plot!(profile_plot, Pz, zw, label = "⟨∂_z P⟩ / max|∂_z P|", color = :red, linewidth = 1.2, alpha = 0.8)

        κᵉᶠᶠ_plot = plot(κᵉᶠᶠ, zw,
                       linewidth = 2,
                          margin = 20pt,
                           label = nothing,
                           xlims = (-1e-2, 1e-1),
                          ylabel = "z (m)",
                          xlabel = "κᵉᶠᶠ (m² s⁻¹)")

        flux_plot = plot(times ./ day, buoyancy_flux_time_series,
                         linewidth = 1,
                            margin = 20pt,
                             label = "Buoyancy flux time series",
                             color = :black,
                             alpha = 0.4,
                            legend = :topright,
                            xlabel = "Time (days)",
                            ylabel = "Buoyancy flux (m² s⁻³)",
                             ylims = (0.0, 1.1 * buoyancy_flux_parameters.maximum_buoyancy_flux))

        plot!(flux_plot, times[1:i] / day, buoyancy_flux_time_series[1:i],
              color = :steelblue,
              linewidth = 6,
              label = nothing)

        scatter!(flux_plot, times[i:i] / day, buoyancy_flux_time_series[i:i],
                 markershape = :rtriangle,
                 color = :steelblue,
                 markerstrokewidth = 0,
                 markersize = 15,
                 label = "Current buoyancy flux")

        t = times[i]
        w_title = @sprintf("w(y = 0 m, t = %-16s) (m s⁻¹)", prettytime(t))
        p_title = @sprintf("P(y = 0 m, t = %-16s) (μM)", prettytime(t))

        # Layout something like:
        #
        # [ w contours ]  [ [⟨P⟩+⟨wP⟩] [κ] ]
        # [ p contours ]  [      Qᵇ(t)     ]
        
        layout = @layout [ Plots.grid(2, 1) [ Plots.grid(1, 2)
                                                     c         ] ]

        plot(w_contours, p_contours, profile_plot, κᵉᶠᶠ_plot, flux_plot,
             title=[w_title p_title "" "" ""],
             layout=layout, size=(1600, 700))
    end

    gif(anim, "convecting_plankton.gif", fps = 8) # hide

finally
    close(fields_file)
    close(averages_file)
end
