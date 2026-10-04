# Conclusion

**Status:** extracted verbatim from the hand-compiled `manuscript.md`
at commit 4539f33 so the build script has a single source of truth.

---

On two traffic clips with hand-annotated trajectories, under the
standard window-level TTC-based SSM label with a corrected cross-
validation split, a single scalar feature matches or beats the trained
models. The failure is not in the model class; it is in the label's
construction. The window-aggregation step, treated as plumbing in the
SSM-ML literature, is where the signal is lost.
