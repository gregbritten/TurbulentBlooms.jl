using Oceananigans, Oceananigans.Utils, Oceananigans.BoundaryConditions, Oceananigans.Grids, Oceananigans.Forcings
using SeawaterPolynomials
using Plots, Random

grid = RegularCartesianGrid(size = (25, 1, 50), extent = (100, 4, 200),topology = (Periodic, Periodic, Bounded),)

buoyancy = SeawaterBuoyancy(
    equation_of_state = SeawaterPolynomials.TEOS10EquationOfState(),
    constant_salinity = 35.0 # psu
)

N² = 1e-5 # s⁻²
α = SeawaterPolynomials.thermal_expansion(20, 35, 0, buoyancy.equation_of_state)
g = buoyancy.gravitational_acceleration
ρᵣ = buoyancy.equation_of_state.reference_density
∂T∂z = ρᵣ * N² / (α * g)
bottom_temperature_boundary_condition = BoundaryCondition(Gradient, ∂T∂z)

peak_outgoing_radiation = 200 # Watts / m²
heat_capacity = 3991 # J / kg / ᵒC
reference_density = buoyancy.equation_of_state.reference_density # kg m⁻³
peak_outgoing_flux = peak_outgoing_radiation / (reference_density * heat_capacity)
Qᵇ = α * g * peak_outgoing_flux

@inline diurnal_cycle(t, day) = max(0, - cos(2π * t / day))
@inline nocturnal_cycle(t, day) = max(0, cos(2π * t / day))
@inline outgoing_flux(x, y, t, p) = p.peak * nocturnal_cycle(t, p.day)

surface_temperature_boundary_condition = BoundaryCondition(Flux, outgoing_flux, parameters=(day=day, peak=peak_outgoing_flux))

T_bcs = TracerBoundaryConditions(grid, bottom = bottom_temperature_boundary_condition,
                                 top = surface_temperature_boundary_condition)

light_attenuation_scale = 20 # m
surface_solar_insolation = 400 # Watts / m²
surface_solar_temperature_flux = surface_solar_insolation / (reference_density * heat_capacity)

@inline daylight(z, t, λ, day) = exp(z / λ) * diurnal_cycle(t, day)
@inline solar_flux_divergence(z, t, Qᴵ, λ, day) = Qᴵ / λ * daylight(z, t, λ, day)
@inline diurnal_solar_flux_divergence(x, y, z, t, p) =
    max(0, solar_flux_divergence(z, t, p.surface_flux, p.attenuation, p.day))

interior_heating = Forcing(diurnal_solar_flux_divergence,
                                 parameters = (surface_flux = surface_solar_temperature_flux,
                                               attenuation = light_attenuation_scale,
                                               day = day))

model = IncompressibleModel(architecture = CPU(),
                                    grid = grid,
                                coriolis = nothing,
                                 tracers = :T, #tracer_names,
                                buoyancy = buoyancy,
                                 closure = IsotropicDiffusivity(ν=1e-3, κ=1e-3),
                     boundary_conditions = (T=T_bcs,),
                                 forcing = (T=interior_heating,),
)

initial_temperature(x, y, z) = (20 + ∂T∂z * z + ∂T∂z * grid.Lz * 1e-4 * randn() * exp(z / (8 * grid.Δz)))

set!(model, T = initial_temperature,)

wizard = TimeStepWizard(cfl=0.2, Δt=1.0, max_change=1.1, max_Δt=20.0)

simulation = Simulation(model, iteration_interval = 1, stop_time=86400, Δt = wizard)

run!(simulation)

Nsimulation = Simulation(model, Δt=10.0, stop_time=86400, iteration_interval=6)

anim = @animate for i in 1:1440*2
    Nsimulation.stop_time += 60
    run!(Nsimulation)

    ## Coordinate arrays for plotting
    xC, zF, zC = xnodes(Cell, grid)[:], znodes(Face, grid)[:], znodes(Cell, grid)[:]

    ## Slices to plots.
    w = Array(interior(model.velocities.w))[:, 1, :]
    T = Array(interior(model.tracers.T))[:, 1, :]

    ## Plot the slices.
    w_plot = heatmap(xC, zF, w', xlabel="x (m)", ylabel="z (m)", color=:balance, clims=(-3e-2, 3e-2))
    T_plot = heatmap(xC, zC, T', xlabel="x (m)", ylabel="z (m)", color=:thermal, clims=(19.75, 20.2))

    ## Arrange the plots side-by-side.
    plot(w_plot, T_plot, layout=(1, 2), size=(1200, 400),
         title=[lpad(i÷1440,2,"0")*"day "*lpad(i÷60-24*(i÷1440),2,"0")*"hour"*" W (m/s)" "temperature (C)"])
end

gif(anim, "stable_mixed_layer.gif", fps = 45) # hide
