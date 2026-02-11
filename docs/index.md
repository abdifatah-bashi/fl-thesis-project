---
hide:
  - navigation
  - toc
  - footer
---

<section class="fl-hero" markdown>
<div class="fl-hero-inner" markdown>

# The model travels, the data stays put.

A **Federated Learning** framework that enables hospitals to collaboratively train heart disease prediction models — without ever sharing a single patient record.

<div class="fl-hero-buttons" markdown>

[Get Started :material-arrow-right:](getting-started.md){ .md-button .md-button--primary .fl-btn-hero }
[Explore Concepts](concepts/index.md){ .md-button .fl-btn-hero }

</div>

</div>
</section>

<section class="fl-features" markdown>
<div class="fl-grid" markdown>

<div class="fl-card animate-on-scroll" markdown>

### :material-shield-lock-outline: Privacy by Design

Patient data **never leaves** the hospital. Only mathematical model parameters are transmitted across the network — raw records stay local.

</div>

<div class="fl-card animate-on-scroll" markdown>

### :material-hospital-building: Cross-Hospital Collaboration

Cleveland Clinic and Hungarian Institute jointly train a shared AI model across institutional boundaries — no data sharing required.

</div>

<div class="fl-card animate-on-scroll" markdown>

### :material-brain: Neural Intelligence

A PyTorch neural network learns heart disease patterns from distributed, heterogeneous data using the Flower framework.

</div>

</div>
</section>

<section class="fl-stats" markdown>
<div class="fl-stats-grid" markdown>

<div class="fl-stat animate-on-scroll" markdown>
<div class="fl-stat-number" data-target="2">0</div>
<div class="fl-stat-label">Hospitals</div>
</div>

<div class="fl-stat animate-on-scroll" markdown>
<div class="fl-stat-number" data-target="590">0</div>
<div class="fl-stat-label">Patients</div>
</div>

<div class="fl-stat animate-on-scroll" markdown>
<div class="fl-stat-number" data-target="13">0</div>
<div class="fl-stat-label">Medical Features</div>
</div>

<div class="fl-stat animate-on-scroll" markdown>
<div class="fl-stat-number" data-target="0">0</div>
<div class="fl-stat-label">Records Shared</div>
</div>

</div>
</section>

<section class="fl-quickstart" markdown>
<div class="fl-quickstart-inner" markdown>

## Run it in 3 commands

<p class="fl-section-sub">From zero to a running federated simulation — in seconds.</p>

```bash
# Install the framework
pip install -e .

# Download the UCI Heart Disease dataset
python scripts/download_data.py

# Launch the full federated simulation
python run_simulation.py
```

[Full setup guide :material-arrow-right:](getting-started.md){ .md-button .md-button--primary }

</div>
</section>
