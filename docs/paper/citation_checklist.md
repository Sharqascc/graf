# Citation verification checklist

Manual pass over the 13 `[CHECK]` entries in `references.md`.

**How to use:** for each entry, open Google Scholar, paste the
search string, look at the top result, and either:

- **Confirm** — the year/venue/pages match what's below → tick `[x]`
- **Correct** — write the correct value in the `FIX:` line
- **Not found** — mark `[?]` and note why (older paper, technical
  report, conference paper not indexed)

---

## 1. Essa & Sayed 2019

**Search:** `Essa Sayed traffic conflict models cycle level intersections`

**Current citation in the paper:**
> Accident Analysis & Prevention, 2019.

**Verify:** volume, page range.

- [ ] confirmed
- FIX:

---

## 2. Hydén 1987

**Search:** `Hyden 1987 Swedish Traffic Conflicts Technique Lund`

**Current citation in the paper:**
> PhD thesis, Lund Institute of Technology, Bulletin 70, 1987.

**Verify:** is it a thesis or a bulletin? Some sources cite both.

- [ ] confirmed
- FIX:

---

## 3. Johnsson, Laureshyn, De Ceunynck 2018

**Search:** `Johnsson Laureshyn De Ceunynck surrogate safety indicators vulnerable road users`

**Current citation in the paper:**
> Transport Reviews, 2018.

**Verify:** volume, pages, exact subtitle.

- [ ] confirmed
- FIX:

---

## 4. Mahmud, Ferreira, Hoque, Tavassoli 2017

**Search:** `Mahmud Ferreira proximal surrogate indicators safety evaluation review`

**Current citation in the paper:**
> IATSS Research, 41(4), 153-163, 2017.

**Verify:** author order, volume/issue/pages.

- [ ] confirmed
- FIX:

---

## 5. Mohamed, Qian, Elhoseiny, Claudel 2020

**Search:** `Social-STGCNN Mohamed Qian Elhoseiny trajectory prediction`

**Current citation in the paper:**
> CVPR 2020, pp. 14424-14432.

**Verify:** page range.

- [ ] confirmed
- FIX:

---

## 6. Perkins & Harris 1968

**Search:** `Perkins Harris traffic conflict characteristics accident potential General Motors`

**Current citation in the paper:**
> Highway Research Record, 225, 35-43. Also SAE Technical Paper 680124.

**Verify:** which one to cite. This is a known dual-citation paper.
Pick one. If you've read the SAE version, cite that; otherwise the
Highway Research Record is standard.

- [ ] confirmed
- FIX:

---

## 7. Schöller, Aravantinos, Lay, Knoll 2020

**Search:** `Scholler Aravantinos constant velocity model pedestrian motion prediction`

**Current citation in the paper:**
> IEEE Robotics and Automation Letters, 2020.

**Verify:** volume, pages.

- [ ] confirmed
- FIX:

---

## 8. Tarko et al. 2009

**Search:** `Tarko surrogate measures of safety TRB annual meeting 2009`

**Current citation in the paper:**
> Transportation Research Board Annual Meeting, White Paper, 2009.

**Verify:** full author list and TRB paper number.

- [ ] confirmed
- FIX:

---

## 9. Wang, Xie, Huang, Liu 2021

**Search:** `Wang Xie Huang Liu surrogate safety measures connected automated vehicles review`

**Current citation in the paper:**
> Accident Analysis & Prevention, 157, 106157, 2021.

**Verify:** author list, volume, article number.

- [ ] confirmed
- FIX:

---

## 10. Yu, Ma, Ren, Zhao 2020

**Search:** `Yu Ma Ren STAR spatio-temporal graph transformer pedestrian trajectory prediction`

**Current citation in the paper:**
> ECCV 2020.

**Verify:** author order.

- [ ] confirmed
- FIX:

---

## 11. Zheng, Ismail, Meng 2014

**Search:** `Zheng Ismail Meng traffic conflict techniques road safety analysis`

**Current citation in the paper:**
> Canadian Journal of Civil Engineering, 41(7), 633-641, 2014.

**Verify:** the venue was corrected from AAP to CJCE in an earlier
commit (PR #86). Confirm volume/pages.

- [ ] confirmed
- FIX:

---

## 12. Bouthillier et al. 2021

**Search:** `Bouthillier accounting for variance machine learning benchmarks MLSys 2021`

**Current citation in the paper:**
> MLSys, 2021.

**Verify:** page range or paper number.

- [ ] confirmed
- FIX:

---

## 13. McDermott et al.

**Already fixed in PR #86.** The citation is now:

> McDermott, M. B. A., et al. (2021). *Reproducibility in machine
> learning for health research: Still a ways to go*. Science
> Translational Medicine, 13(586).

**Optional:** verify volume/issue. Low priority.

- [ ] confirmed
- FIX:

---

## When done

Paste the checklist back, or just tell me which entries had
`FIX:` values. I'll update `references.md` and push in one commit.

## Separately: recent references

After the checks, the paper needs 20-30 citations from 2024-2026.

**Search terms that will surface recent work:**

- `surrogate safety measures machine learning 2024 2025`
- `traffic conflict prediction deep learning 2025`
- `time to collision estimation video 2024`
- `window label classification traffic safety`
- `autonomous vehicle safety validation surrogate 2025`

For each paper you find that's relevant, note: authors, year,
title, venue, DOI. Same format as `references.md`. 20-30 entries.
