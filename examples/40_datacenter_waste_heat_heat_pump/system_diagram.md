# Data center + heat pump + district heating — system diagram

Interconnected systems for [run_datacenter_waste_heat_heat_pump.py](run_datacenter_waste_heat_heat_pump.py).

## Data flow

```mermaid
flowchart LR
    classDef feedstock fill:#e6f2ff,stroke:#3366cc,color:#000
    classDef grid fill:#fff2cc,stroke:#b58900,color:#000
    classDef converter fill:#e8f5e9,stroke:#2e7d32,color:#000
    classDef demand fill:#fde7f3,stroke:#ad1457,color:#000

    WF["water_feedstock<br/><i>FeedstockPerformanceModel</i><br/>rated_capacity: 1e6 galUS/h"]:::feedstock
    GB["grid_buy<br/><i>GridPerformanceModel</i><br/>interconnection: 1e6 kW<br/>buy price: $0.04574/kWh"]:::grid
    GBH["grid_buy_hp<br/><i>GridPerformanceModel</i><br/>interconnection: 1e5 kW<br/>buy price: $0.04574/kWh"]:::grid

    DC["<b>data_center</b><br/><i>DataCenterPUEWUEPerformanceModel</i><br/>IT workload: 100 MW<br/>PUE 1.4 / WUE 1.0<br/>cooling case 9 (air-cooled)<br/>waste-heat fraction 0.06<br/>supply/return 28 / 20 °C"]:::converter
    HP["<b>heat_pump</b><br/><i>HeatPumpPerformanceModel</i><br/>mode: carnot<br/>cap: 15 MW_th<br/>delivery: 75 °C<br/>η_carnot: 0.5"]:::converter
    DH["<b>district_heating</b><br/><i>DistrictHeatingDemand</i><br/>demand: 12 MW_th<br/>min supply: 70 °C<br/>capacity: 20 MW_th"]:::demand

    %% Water loop
    WF -- "water_out → water_in<br/>(galUS/h)" --> DC

    %% Electricity to data center
    DC -- "unmet_electricity_demand →<br/>electricity_set_point" --> GB

    %% Waste-heat handoff to HP
    DC -- "waste_heat_out → heat_in<br/>(MW_th)" --> HP
    DC -- "waste_heat_supply_temp_C →<br/>heat_supply_temp_C_in (°C)" --> HP

    %% Electricity to heat pump
    HP -- "electricity_demand →<br/>electricity_set_point" --> GBH
    GBH -- "electricity_out → electricity_in<br/>(kW)" --> HP

    %% Upgraded heat to DH
    HP -- "heat_out → heat_in<br/>(MW_th)" --> DH
    HP -- "heat_supply_temp_C →<br/>heat_supply_temp_C_in (°C)" --> DH
```

## Connection table

Raw entries from `technology_interconnections` in [plant_config.yaml](plant_config.yaml).

| Source | Source signal | Sink | Sink signal | Kind |
|---|---|---|---|---|
| `data_center` | `unmet_electricity_demand` | `grid_buy` | `electricity_set_point` | signal |
| `water_feedstock` | `water` | `data_center` | `water` | pipe (auto) |
| `data_center` | `waste_heat_out` | `heat_pump` | `heat_in` | signal |
| `data_center` | `waste_heat_supply_temp_C` | `heat_pump` | `heat_supply_temp_C_in` | signal |
| `grid_buy_hp` | `electricity` | `heat_pump` | `electricity` | cable (auto) |
| `heat_pump` | `electricity_demand` | `grid_buy_hp` | `electricity_set_point` | signal |
| `heat_pump` | `heat` | `district_heating` | `heat` | heat_transport (auto) |
| `heat_pump` | `heat_supply_temp_C` | `district_heating` | `heat_supply_temp_C_in` | signal |

## Operating-point numbers (from a full-year run)

| Quantity | Value |
|---|---|
| DC IT power → total facility power | 100 → 140 MW |
| Waste heat available (source) | ≈ 73,584 MWh/yr (8.4 MW × 8760 h) |
| HP electricity used | ≈ 27,216 MWh/yr |
| HP heat delivered @ 75 °C | ≈ 100,800 MWh/yr (COP ≈ 3.70) |
| DH demand met / unmet | 100,800 / 4,320 MWh/yr |
