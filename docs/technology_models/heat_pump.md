# Heat Pump Model

The heat pump model, implemented in {mod}`h2integrate.converters.heat.heat_pump`, upgrades
low-grade heat (for example, data center waste heat) to a higher delivery temperature by
consuming electricity. It produces the `heat` commodity (in MW).

| Model | Type | Description |
| --- | --- | --- |
| `HeatPumpPerformanceModel` | Performance | Heat delivered, electricity used, and COP |
| `HeatPumpCostModel` | Cost | CapEx, fixed OpEx, variable OpEx, optional interconnection pipe costs, and optional heat-sale revenue |

## Performance

The heat pump delivers heat at `delivery_temp_C` from a source heat stream (`heat_in`) at
`heat_supply_temp_C_in`. Two modes are available through `hp_mode`:

- `fixed_cop`: a constant coefficient of performance, `cop`.
- `carnot`: a temperature-dependent COP calculated from a second-law efficiency
  $\eta_{\text{carnot}}$ (`carnot_efficiency`) and the source and sink temperatures in Kelvin:

```{math}
\text{COP} = \eta_{\text{carnot}} \frac{T_{\text{sink}}}{T_{\text{sink}} - T_{\text{source}}}
```

The Carnot COP is clipped between 1 and 20. In Carnot mode, no heat is delivered if the
source temperature is at or above the delivery temperature, or below `min_source_temp_C`
(if set).

From the energy balance $Q_{\text{delivered}} = Q_{\text{source}} + W_{\text{elec}}$, the
heat pump draws a fraction $1 - 1/\text{COP}$ of its delivered heat from the source and
$1/\text{COP}$ from electricity. Heat output at each timestep is limited by:

1. the rated capacity, `system_capacity_mw_th`
2. the available source heat, $Q_{\text{source}} / (1 - 1/\text{COP})$
3. the available electricity, $W_{\text{elec}} \times \text{COP}$

An optional off-taker demand ceiling (`heat_demand_profile`) can also limit output. When
`throttle_to_demand` is `true` (the default), the heat pump only produces heat the
off-taker can absorb. When `false`, it runs at its supply-limited potential and the excess
is reported as `heat_curtailed`.

Key outputs:

| Output | Description |
| --- | --- |
| `heat_out` | Heat produced (MW) |
| `heat_supply_temp_C` | Delivery temperature (degC) |
| `electricity_used` | Electricity consumed (MW) |
| `electricity_demand` | Electricity needed for the capacity-, source-, and demand-limited output, independent of `electricity_in` (MW). Use this to set a grid purchase. |
| `cop_actual` | COP at each timestep |
| `heat_delivered`, `heat_curtailed` | Heat delivered to, and curtailed by, the off-taker (MW) |

## Cost

`HeatPumpCostModel` calculates:

- CapEx: `capex_per_mw_th` $\times$ `system_capacity_mw_th`, plus the interconnection pipe
  CapEx
- Fixed OpEx: `fixed_opex_per_mw_th_per_year` $\times$ `system_capacity_mw_th`, plus the
  interconnection pipe O&M (`dh_pipe_annual_om_fraction` $\times$ pipe CapEx)
- Variable OpEx: `variable_opex_per_mwh_th` $\times$ annual delivered heat, less any
  heat-sale revenue (`heat_sell_price_usd_per_mwh_th`)

The optional buried interconnection pipe to the off-taker has length
`dh_connection_distance_m`. Its diameter is sized as
$D_{\text{mm}} = c \sqrt{Q_{\text{cap}}}$, where $c$ is
`dh_pipe_diameter_coeff_mm_per_sqrt_mw`. The pipe cost per meter is either
`dh_pipe_fixed_cost_usd_per_m` + `dh_pipe_cost_per_mm_diameter_usd_per_m` $\times D_{\text{mm}}$,
or a bulk `dh_trench_bulk_cost_usd_per_ft` if provided.

## Example

`examples/40_datacenter_waste_heat_heat_pump/` upgrades 28 degC waste heat from an
air-cooled data center to 75 degC for a district-heating loop with a 70 degC minimum supply
temperature. A separate grid connection supplies the heat pump's electricity.

```yaml
heat_pump:
  performance_model:
    model: HeatPumpPerformanceModel
  cost_model:
    model: HeatPumpCostModel
  model_inputs:
    shared_parameters:
      system_capacity_mw_th: 15.0
    performance_parameters:
      hp_mode: carnot
      delivery_temp_C: 75.0
      carnot_efficiency: 0.5
      min_source_temp_C: 5.0
    cost_parameters:
      cost_year: 2026
      capex_per_mw_th: 8.0e+5
      fixed_opex_per_mw_th_per_year: 2.0e+4
      variable_opex_per_mwh_th: 1.0
```
