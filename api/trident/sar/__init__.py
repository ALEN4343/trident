"""SAR oil-spill segmentation.

A trained DeepLabv3+ operating on Sentinel-1 and PALSAR backscatter, carried
over from SAMUDRA. This is a genuinely different sensing modality from the rest
of TRIDENT, not a second opinion on the same pixels: the optical branch reads
an interference film in visible light, this reads the way oil flattens the sea
surface and kills radar backscatter. They are kept apart because feeding an
optical photograph to a model trained on SAR backscatter produces confident
nonsense.
"""

from .segment import SARSegmenter, checkpoint_status, polygons, segment_array

__all__ = ["SARSegmenter", "checkpoint_status", "polygons", "segment_array"]
