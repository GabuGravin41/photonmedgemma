# Motivation: Why Photonic AI for Medicine?

## The Crisis of AI Infrastructure

Artificial intelligence has made extraordinary progress. Large language models can now assist with medical diagnosis, analyze radiological images, and generate treatment plans that rival specialist physicians. Yet this progress comes at a cost that is fundamentally unsustainable:

- A single training run for a frontier model can consume **1,000+ MWh** of electricity — equivalent to the annual energy consumption of 100 American homes.
- Running inference on GPT-4 costs approximately **$0.002–0.06 per query**, which scales to millions of dollars per day at production loads.
- Data center cooling alone accounts for 30–40% of total energy consumption.
- Global AI compute demand is doubling every 6–12 months, outpacing renewable energy deployment by a wide margin.

The leading AI companies — Google, Microsoft, OpenAI, Meta — have collectively committed hundreds of billions of dollars to GPU infrastructure. They are, in effect, betting that a technological breakthrough in compute efficiency will arrive before their capital runs out. That breakthrough is photonics.

## The Medical AI Access Problem

Beyond sustainability, there is a more immediate human problem: AI-powered medical intelligence is unavailable where it is needed most.

In 2024, we worked directly with neurosurgeons at **Kenyatta National Hospital (KNH)** in Nairobi, Kenya — the largest referral hospital in East and Central Africa. The neurosurgery department serves a catchment of tens of thousands of Kenyans with a fraction of the specialists required. The needs were clear:

1. **Smart triage**: automatically prioritizing patients with life-threatening conditions (cerebral hemorrhage, herniation) from CT scans.
2. **Diagnostic assistance**: AI-assisted differential diagnosis for complex neurological presentations.
3. **Treatment planning**: generating evidence-based management protocols.

MedGemma — Google's 4-billion-parameter multimodal medical AI — can perform all of these tasks with clinical-grade accuracy. But deploying it at KNH posed insurmountable barriers:

| Barrier | Detail |
|---------|--------|
| **Data sovereignty** | Kenyan law and hospital policy prohibit sending patient imaging data to foreign servers |
| **Connectivity** | Hospital internet connectivity is insufficient for real-time inference |
| **Cost** | Cloud AI API costs are prohibitive at the budget available |
| **Latency** | Emergency triage requires sub-second decision support |

The answer is not a smaller model. The answer is **different hardware**.

## Why Photonics?

A photonic chip performs computation using light rather than electrical charge. The physics of light gives photonics several properties that make it ideal for neural network inference:

### Speed
Light travels at ~200,000 km/s in silicon waveguides. A photonic matrix multiplication is completed in the time it takes light to traverse the chip — nanoseconds for a chip the size of a fingernail. This is fundamentally faster than electronic compute, which is limited by RC time constants, clock distribution, and pipeline filling.

### Energy
Moving an electron requires energy. Moving a photon is free — light propagates through a waveguide with essentially zero energy cost. The only energy costs in a photonic chip are:
- The laser source (shared across all computations)
- Phase shifters (set once at chip configuration; near-zero power in static mode)
- Photodetectors (read-out)

Practical photonic neural network demonstrations have achieved **<1 pJ per multiply-accumulate operation**, compared to 1–10 pJ/MAC for the best electronic accelerators (TPUs, H100). For a 4B parameter model performing a forward pass, this translates to orders-of-magnitude energy savings.

### Parallelism
A single optical waveguide can carry many different signals simultaneously on different wavelengths (wavelength-division multiplexing, WDM). A photonic chip with 64 wavelengths effectively gets 64× parallelism for free, with no additional power cost for the parallelism itself.

### Thermal Behavior
Electronic chips generate enormous heat from switching currents. Photonic chips — especially in **static weight** mode where phase shifters are set once — generate minimal heat. This eliminates the need for active cooling, making deployment in resource-limited settings (rural hospitals, field clinics, ambulances) practical.

## The Static Compilation Insight

The key insight of this project is that **inference from a fixed, trained model is a static operation**.

MedGemma's weights are fixed after training. They do not change during inference. This means:

1. We do not need *programmable* hardware in the traditional sense.
2. We can **encode the weights permanently** into the physical structure of the chip.
3. Inference then becomes pure physics — light goes in, light comes out, and the computation has been done by the chip's geometry itself.

This is analogous to an ASIC (Application-Specific Integrated Circuit) versus a GPU. A GPU is flexible but inefficient. An ASIC is optimized for one task and orders of magnitude more efficient. We are building the photonic equivalent of an ASIC for MedGemma inference.

In electronics, this is called *spatial computing* or *in-memory computing*. In photonics, it is sometimes called *diffractive deep learning* or *optical analog computing*. We call it **static photonic compilation**.

## The Path Forward

This project does not promise a fabricated chip today. Tape-out (manufacturing) requires access to a photonic foundry (imec, GlobalFoundries, AMF, LIGENTEC) and significant capital. What this project delivers is:

1. **The complete compiler**: software that takes MedGemma's weights and produces photonic circuit specifications (phase angles for every MZI on the chip).
2. **The chip architecture**: a complete physical design (GDS layout files) ready for foundry submission.
3. **The theoretical foundation**: analysis of accuracy, energy, and throughput trade-offs.
4. **The open-source framework**: so that any researcher, hospital, or government can replicate and extend this work.

The goal is that within 2–3 years, a photonic MedGemma chip can be manufactured for under $100/unit at volume, powered from a USB-C port, and deployed in every major hospital in Africa, Southeast Asia, and the global south — running world-class medical AI inference with no cloud dependency, no data privacy risk, and no energy crisis.

This is the future we are building.
