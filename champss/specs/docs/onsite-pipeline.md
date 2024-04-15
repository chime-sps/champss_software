# Onsite Pipeline

```
┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐
│  L1 data   │     │ Impulsive  │     │Beamforming │     │Periodicity │     │ Candidate  │
│downsampled ├────▶│    RFI     │────▶│     +      │────▶│  search &  │────▶│ processing │
│   stream   │     │  Excision  │     │dedispersion│     │  stacking  │     │            │
└────────────┘     └────────────┘     └────────────┘     └────────────┘     └────────────┘
```

## Periodicity Search
```
                                      Periodicity statistics─────┐
                                      │                          │
                                      │      ┌────────────┐      │
                                      │      │   Power    │      │
                                      │ ┌───▶│  Spectrum  │────┐ │
                                      │ │    │Computation │    │ │
   ┌ ─ ─ ─ ─ ─ ─      ┌────────────┐  │ │    └────────────┘    │ │  ┌────────────┐                        ┌ ─ ─ ─ ─ ─ ─
    Dedispersed │     │Periodic RFI│  │ │                      │ │  │ Single-ptg │                          Candidate  │
   │time-stream  ────▶│  removal   │──┼─┤                      ├─┼─▶│ candidate  │──┬──────────────────┬─▶│ processing
                │     │            │  │ │                      │ │  │   search   │  │                  │               │
   └ ─ ─ ─ ─ ─ ─      └────────────┘  │ │                      │ │  └────────────┘  │                  │  └ ─ ─ ─ ─ ─ ─
                                      │ │    ┌────────────┐    │ │                  │  ┌────────────┐  │
                                      │ │    │   H-Hat    │    │ │                  │  │ Stacking & │  │
                                      │ └───▶│ Statistics │────┘ │                  └─▶│  multiday  │──┘
                                      │      │Computation │      │                     │   search   │
                                      │      └────────────┘      │                     └────────────┘
                                      │                          │
                                      └──────────────────────────┘
```


## Candidate Processing
```
                                                                               ┌ ─ ─ ─ ─ ─ ─
                                                                                Periodic RFI│    Update
                                                                               │  removal    ◀──birdies ─
                                                                                            │     list   │
                                                                               └ ─ ─ ─ ─ ─ ─             │
   ┌ ─ ─ ─ ─ ─ ─      ┌────────────┐     ┌────────────┐
      Single-   │     │ Candidate  │     │  Feature   │                                                  │
   │  pointing   ────▶│ clustering │────▶│ generation │──┐                                               │
     candidates │     │            │     │            │  │
   └ ─ ─ ─ ─ ─ ─      └────────────┘     └────────────┘  │                                               │
                                                         │                                               │
   ┌ ─ ─ ─ ─ ─ ─      ┌────────────┐     ┌────────────┐  │  ┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐     ┌ ─ ─ ─ ─ ─ ─
      Single-   │     │ Candidate  │     │  Feature   │  │  │  Grouping  │     │ Position-  │     │ Candidate  │     │Known-source│       Candidate  │
   │  pointing   ────▶│ clustering │────▶│ generation │──┼─▶│  between   │────▶│   based    │────▶│classificati│────▶│   sifter   │────▶│verification
     candidates │     │            │     │            │  │  │ pointings  │     │  features  │     │     on     │     │            │                  │
   └ ─ ─ ─ ─ ─ ─      └────────────┘     └────────────┘  │  └────────────┘     └────────────┘     └────────────┘     └────────────┘     └ ─ ─ ─ ─ ─ ─
                                    ...                  │
   ┌ ─ ─ ─ ─ ─ ─      ┌────────────┐     ┌────────────┐  │
      Single-   │     │ Candidate  │     │  Feature   │  │
   │  pointing   ────▶│ clustering │────▶│ generation │──┘
     candidates │     │            │     │            │
   └ ─ ─ ─ ─ ─ ─      └────────────┘     └────────────┘
```
