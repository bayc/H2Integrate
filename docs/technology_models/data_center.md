# Data Center Models

H2Integrate includes two data center model pairs, implemented in
{mod}`h2integrate.converters.data_center`. Both produce a `compute_load` commodity (in MW)
and calculate the electricity and water the data center consumes to serve a compute
load profile. An example that runs both model pairs with an hourly AI training workload
is provided in `examples/38_data_center/`.

| Model | Type | Description |
| --- | --- | --- |
| `DataCenterPerformanceModel` | Performance | Electricity demand from a compute electrical efficiency and a cooling load ratio |
| `DataCenterCostModel` | Cost | CapEx, fixed OpEx, and variable OpEx per MWh of compute load |
| `DataCenterPUEWUEPerformanceModel` | Performance | Electricity and water demand from Power Usage Effectiveness (PUE) and Water Usage Effectiveness (WUE) |
| `DataCenterPUEWUECostModel` | Cost | CapEx, fixed OpEx, and electricity and water costs |

## Simple Data Center Model

### Performance

`DataCenterPerformanceModel` caps the compute load demand (`compute_load_demand`) at the
data center capacity (`system_capacity_mw`). The total electricity demand is the compute
load scaled by the compute electrical efficiency, plus a cooling load proportional to the
electrical compute load:

```{math}
P_{\text{elec}} = \frac{P_{\text{compute}}}{\eta_{\text{compute}}} \left(1 + r_{\text{cooling}}\right)
```

where $\eta_{\text{compute}}$ is `compute_electrical_efficiency` and $r_{\text{cooling}}$ is
`cooling_load_ratio`. Water demand is the compute load multiplied by
`water_use_gal_per_mwh`, limited by the available `water_in`.

Any electricity demand that is not met by `electricity_in` is reported as
`unmet_electricity_demand_out`, which can be connected to a grid model's
`electricity_set_point` to purchase the required electricity.

### Cost

`DataCenterCostModel` calculates:

- CapEx: `capex_per_mw` $\times$ `system_capacity_mw`
- Fixed OpEx: `fixed_opex_per_mw_per_year` $\times$ `system_capacity_mw`
- Variable OpEx: `variable_opex_per_mwh` $\times$ total compute load (MWh)

## PUE/WUE Data Center Model

### Performance

`DataCenterPUEWUEPerformanceModel` uses industry-standard efficiency metrics to determine
facility power and water use from a compute IT workload (`compute_it_workload`):

```{math}
P_{\text{facility}} = P_{\text{IT}} \times \text{PUE}
```

```{math}
\dot{V}_{\text{water}} = E_{\text{IT}} \times \text{WUE}
```

where PUE is the ratio of total facility power to IT equipment power, and WUE is the water
consumed in liters per kWh of IT energy.

PUE and WUE can be provided in one of three ways:

1. **Directly**: supply both `pue` and `wue`.
2. **By climate zone**: supply an IECC `climate_zone` (for example, `"4A"`), and PUE/WUE
   are looked up from the climate- and technology-specific estimates of
   Lei and Masanet (2022) [^lei2022].
3. **By site location**: supply none of the above, and the IECC climate zone is determined
   from the site `latitude` and `longitude` in the plant configuration using the
   [`eeweather`](https://github.com/openeemeter/eeweather) package.

For options 2 and 3, `efficiency_level` selects the 5th-percentile (`"efficient"`) or
95th-percentile (`"inefficient"`) estimates. The cooling configuration can be set with
`cooling_configuration`:

| Case | Size category | Cooling configuration |
| --- | --- | --- |
| 1 | Large-scale | Airside economizer + adiabatic cooling + (water-cooled chiller) |
| 2 | Large-scale | Waterside economizer + (water-cooled chiller) |
| 3 | Midsize | Airside economizer + (water-cooled chiller) |
| 4 | Midsize | Waterside economizer + (water-cooled chiller) |
| 5 | Midsize | Water-cooled chiller |
| 6 | Midsize | Airside economizer + (air-cooled chiller) |
| 7 | Midsize | Air-cooled chiller |
| 8 | Small | Water-cooled chiller |
| 9 | Small | Air-cooled chiller |
| 10 | Small | Direct expansion system |
| 11 | Large-scale or midsize | Direct-to-chip liquid loop + dry cooling + adiabatic cooling + (air-cooled chiller) |
| 12 | Large-scale or midsize | Direct-to-chip liquid loop + waterside economizer + (water-cooled chiller) |

If `cooling_configuration` is not set, the configuration with the lowest PUE (or lowest WUE
when `optimize_for: wue`) is selected automatically. Providing `size_sqft` restricts the
automatic selection to configurations valid for that size category: large-scale
(> 20,000 sqft), midsize (1,000 - 20,000 sqft), or small (< 1,000 sqft).

Unmet electricity and water demands are reported as `unmet_electricity_demand` and
`unmet_water_demand`.

### Waste Heat Recovery

The PUE/WUE performance model also reports the recoverable waste heat as a fraction of total
facility power:

```{math}
Q_{\text{waste}} = f_{\text{recoverable}} \times P_{\text{facility}}
```

The recoverable fraction and the waste-heat supply and return temperatures default to
literature-based values for each cooling configuration:

| Case | Recoverable fraction | Supply temp (°C) | Return temp (°C) |
| --- | --- | --- | --- |
| 1 | 0.10 | 30 | 20 |
| 2 | 0.15 | 35 | 25 |
| 3 | 0.10 | 30 | 20 |
| 4 | 0.15 | 35 | 25 |
| 5 | 0.20 | 40 | 30 |
| 6 | 0.08 | 28 | 20 |
| 7 | 0.08 | 30 | 20 |
| 8 | 0.12 | 35 | 25 |
| 9 | 0.06 | 28 | 20 |
| 10 | 0.05 | 30 | 22 |
| 11 | 0.45 | 45 | 35 |
| 12 | 0.55 | 50 | 40 |

These defaults are order-of-magnitude estimates informed by the literature:

- **Air-cooled, chiller-based, and direct-expansion configurations (cases 1-10):** recover
  5-20% of facility power at 28-40 °C. Water-cooled loops reject heat at higher
  temperatures and recover more than air-cooled or direct-expansion systems. Small
  facilities generally recover less because their piping is more distributed. These values
  are based on reviews of data center cooling technology, low-grade waste heat recovery,
  and data center waste heat reuse in district heating
  [^ebrahimi2014] [^wahlroos2017] [^huang2020].
- **Direct-to-chip liquid cooling (cases 11-12):** recovers 45-55% of facility power at
  45-50 °C. Warm-water loops in direct contact with the chips capture a much larger share of
  IT power as higher-grade heat. These values are based on measurements of hot-water-cooled
  and chiller-less liquid-cooled data centers [^zimmermann2012] [^iyengar2012].

The supply temperatures in cases 1-10 are generally too low for conventional district
heating networks, so the heat is typically upgraded with a [heat pump](heat_pump.md).

Each value can be overridden with `waste_heat_recoverable_fraction`,
`waste_heat_supply_temp_C`, and `waste_heat_return_temp_C`. If the cooling configuration is
neither set nor determined from the climate zone (for example, when `pue` and `wue` are
provided directly), all three overrides must be provided.

The outputs `waste_heat_out` (MW), `waste_heat_supply_temp_C`, `waste_heat_return_temp_C`,
`total_waste_heat_recovered`, and `annual_waste_heat_recovered` can be connected to
downstream heat consumers, such as a [heat pump](heat_pump.md) or a
[district heating demand](../demand/demand_components.md), for example:

```yaml
technology_interconnections:
  - [data_center, district_heating, [waste_heat_out, heat_in]]
  - [data_center, district_heating, [waste_heat_supply_temp_C, heat_supply_temp_C_in]]
```

See `examples/39_datacenter_waste_heat_direct/` and
`examples/40_datacenter_waste_heat_heat_pump/`.

### Cost

`DataCenterPUEWUECostModel` calculates:

- CapEx: `capex_per_mw` $\times$ `system_capacity_mw`
- Fixed OpEx: `fixed_opex_per_mw_per_year` $\times$ `system_capacity_mw`
- Electricity cost: `electricity_rate` $\times$ total facility energy (kWh)
- Water cost: `water_rate` $\times$ total water consumed (gal)
- Waste-heat revenue (subtracted from OpEx): `waste_heat_sale_price_usd_per_mwh` $\times$
  total recoverable waste heat (MWh). Defaults to 0.

If electricity and water are purchased through separate technologies (for example, a grid
model and a water feedstock), set `electricity_rate` and `water_rate` to zero to avoid
counting those costs twice.

[^lei2022]: Lei, N., and Masanet, E. "Climate- and technology-specific PUE and WUE
estimations for U.S. data centers using a hybrid statistical and thermodynamics-based
approach." *Resources, Conservation and Recycling* 182 (2022): 106323.

[^ebrahimi2014]: Ebrahimi, K., Jones, G. F., and Fleischer, A. S. "A review of data center
cooling technology, operating conditions and the corresponding low-grade waste heat recovery
opportunities." *Renewable and Sustainable Energy Reviews* 31 (2014): 622-638.
https://doi.org/10.1016/j.rser.2013.12.007

[^wahlroos2017]: Wahlroos, M., Pärssinen, M., Manner, J., and Syri, S. "Utilizing data center
waste heat in district heating - Impacts on energy efficiency and prospects for
low-temperature district heating networks." *Energy* 140 (2017): 1228-1238.
https://doi.org/10.1016/j.energy.2017.08.078

[^huang2020]: Huang, P., Copertaro, B., Zhang, X., et al. "A review of data centers as
prosumers in district energy systems: Renewable energy integration and waste heat reuse for
district heating." *Applied Energy* 258 (2020): 114109.
https://doi.org/10.1016/j.apenergy.2019.114109

[^zimmermann2012]: Zimmermann, S., Meijer, I., Tiwari, M. K., Paredes, S., Michel, B., and
Poulikakos, D. "Aquasar: A hot water cooled data center with direct energy reuse."
*Energy* 43 (2012): 237-245. https://doi.org/10.1016/j.energy.2012.04.037

[^iyengar2012]: Iyengar, M., David, M., Parida, P., et al. "Server liquid cooling with
chiller-less data center design to enable significant energy savings." In *2012 28th Annual
IEEE Semiconductor Thermal Measurement and Management Symposium (SEMI-THERM)* (2012):
212-223. https://doi.org/10.1109/STHERM.2012.6188851
