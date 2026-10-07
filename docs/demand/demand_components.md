# Demand Components

Demand components define rule-based logic for meeting commodity demand profiles without using dynamic system feedback. These components operate independently at each timestep.

This page documents three demand types:
1. [`GenericDemandComponent`](#generic-demand-component) — meets a fixed demand profile.
2. [`FlexibleDemandComponent`](#flexible-demand-component) — adjusts demand up or down within flexible bounds.
3. [`DistrictHeatingDemand`](#district-heating-demand-component) — meets a heat demand profile only when the supply temperature is high enough.


(demand-component-inputs-and-outputs)=
## Demand component inputs and outputs
The inputs to the demand components are the `{commodity}_demand` profile and the `{commodity}_in` profile. The `{commodity}_in` profile is the initial commodity used to satisfy the demand. Suppose the `{commodity}_in` (pink) and `{commodity}_demand` (green) profiles look like whats shown below:

![](./figures/demand_inputs.png)

The demand components then compute the following output profiles:
- `unmet_{commodity}_demand_out`: Unmet demand (non-zero when supply < demand, otherwise 0.)
- `unused_{commodity}_out`: Unused commodity (non-zero when supply > demand, otherwise 0.)

![](./figures/demand_calcs.png)

- `commodity_out`: Delivered output (commodity supplied to demand sink)

![](./figures/demand_commodity_out.png)

The demand components also compute the following outputs:
- `capacity_factor`: the ratio of the demand that's been met to the full demand
- `rated_commodity_production`: the maximum value of the demand profile


(generic-demand-component)=
### Generic Demand Component
The `GenericDemandComponent` allocates commodity input to meet a defined demand profile. It does not contain energy storage logic, only **instantaneous** matching of supply and demand.

The demand component computes each value per timestep:
- Unmet demand (non-zero when supply < demand, otherwise 0.)
- Unused commodity (non-zero when supply > demand, otherwise 0.)
- Delivered output (commodity supplied to demand sink)

This provides a simple baseline for understanding supply–demand balance.

#### Configuration
The demand is defined within the `tech_config` and requires these inputs.

| Field             | Type           | Description                           |
| ----------------- | -------------- | ------------------------------------- |
| `commodity_name`  | `str`          | Commodity name (e.g., `"hydrogen"`).  |
| `commodity_units` | `str`          | Units (e.g., `"kg/h"`).               |
| `demand_profile`  | scalar or list | Timeseries demand or constant demand. |

```yaml
performance_model:
    model: GenericDemandComponent
model_inputs:
  performance_parameters:
    commodity_name: hydrogen
    commodity_units: kg/h
    demand_profile: [10, 10, 12, 15, 14]
```
For an example of how to use the `GenericDemandComponent` framework, see the following:
- `examples/23_solar_wind_ng_demand`

(flexible-demand-component)=
### Flexible Demand Component
The `FlexibleDemandComponent` extends the generic demand component by allowing the actual demand to flex up or down within defined bounds. This is useful for demand-side management scenarios where:
- Processes can defer demand (e.g., flexible industrial loads)
- The system requires demand elasticity without dynamic optimization

The component computes:
- Flexible demand (clamped within allowable ranges)
- Unmet flexible demand
- Unused commodity
- Delivered output

For an example of how to use the `FlexibleDemandComponent` demand component, see the following:
- `examples/23_solar_wind_ng_demand`

The flexible demand component takes an input commodity production profile, the maximum demand profile, and various constraints (listed below), and creates a "flexible demand profile" that follows the original input commodity production profile while satisfying varying constraint.
Please see the figure below for an example of how the flexible demand profile can vary from the original demand profile based on the input commodity production profile and the ramp rates.
The axes are unlabeled to allow for generalization to any commodity and unit type.

| ![Flexible Demand Example](figures/flex_demand_fig.png) |
|-|


#### Configuration
The flexible demand component is defined within the `tech_config` with the following parameters:

| Field               | Type           | Description                                  |
| ------------------- | -------------- | -------------------------------------------- |
| `commodity_name`          | `str`          | Commodity name.                              |
| `commodity_units`         | `str`          | Units for all values.                        |
| `demand_profile`          | scalar or list | Default (nominal) demand profile.            |
| `turndown_ratio`          | float          | Minimum fraction of baseline demand allowed. |
| `ramp_down_rate_fraction` | float          | Maximum ramp-down rate per timestep expressed as a fraction of baseline demand. |
| `ramp_up_rate_fraction` | float          | Maximum ramp-up rate per timestep expressed as a fraction of baseline demand. |
| `min_utilization` | float          | Minimum total fraction of baseline demand that must be met over the entire simulation. |

```yaml
model_inputs:
  performance_parameters:
    commodity_name: hydrogen
    commodity_units: kg/h
    demand_profile: [10, 12, 10, 8]
    turndown_ratio: 0.1
    ramp_down_rate_fraction: 0.5
    ramp_up_rate_fraction: 0.5
    min_utilization: 0
```

(district-heating-demand-component)=
### District Heating Demand Component
The `DistrictHeatingDemand` component behaves like the `GenericDemandComponent` for the `heat`
commodity, and also checks the incoming supply temperature (`heat_supply_temp_C_in`) against
the minimum supply temperature of the district-heating loop (`min_supply_temp_C`). When
`strict_temperature` is `true` (the default), heat supplied below the minimum temperature is
rejected: `heat_out` is zero and the full demand is reported as unmet.

In addition to the standard demand outputs, the component reports:
- `temperature_shortfall_flag`: 1 if the supply temperature is below the minimum, otherwise 0
- `temperature_shortfall_hours`: simulated hours during which the supply temperature is below the minimum

The companion `DistrictHeatingDemandCostModel` calculates CapEx and fixed OpEx per MW of
connected thermal capacity, plus a variable OpEx per MWh of delivered heat.

#### Configuration

| Field                           | Type           | Description                                                         |
| ------------------------------- | -------------- | ------------------------------------------------------------------- |
| `commodity`                     | `str`          | Commodity name; use `heat`.                                         |
| `commodity_rate_units`          | `str`          | Units of the demand profile (e.g., `MW`).                           |
| `demand_profile`                | scalar or list | Timeseries or constant heat demand.                                 |
| `min_supply_temp_C`             | float          | Minimum acceptable supply temperature in degC.                      |
| `system_capacity_mw_th`         | float          | Connected thermal capacity in MW.                                   |
| `strict_temperature`            | bool           | Reject heat supplied below `min_supply_temp_C`. Defaults to `true`. |
| `capex_per_mw_th`               | float          | Capital cost per MW of connected thermal capacity (cost model).     |
| `fixed_opex_per_mw_th_per_year` | float          | Fixed O&M per MW of connected thermal capacity (cost model).        |
| `variable_opex_per_mwh_th`      | float          | Variable O&M per delivered MWh (cost model). Defaults to 0.         |

```yaml
district_heating:
  performance_model:
    model: DistrictHeatingDemand
  cost_model:
    model: DistrictHeatingDemandCostModel
  model_inputs:
    shared_parameters:
      system_capacity_mw_th: 40.0
    performance_parameters:
      commodity: heat
      commodity_rate_units: MW
      demand_profile: 30.0
      min_supply_temp_C: 35.0
      strict_temperature: true
    cost_parameters:
      cost_year: 2026
      capex_per_mw_th: 5.0e+5
      fixed_opex_per_mw_th_per_year: 1.5e+4
      variable_opex_per_mwh_th: 0.5
```

For examples of how to use the `DistrictHeatingDemand` component, see the following:
- `examples/39_datacenter_waste_heat_direct`
- `examples/40_datacenter_waste_heat_heat_pump`
