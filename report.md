# NextGen Health Agent: synthetic-data run report

> Synthetic data and placeholder thresholds. Population prioritisation only, not a diagnosis. A clinician makes every individual decision.

## Headline numbers
- People processed: **3000**
- Risk tiers: Elevated: 355, High: 90, Insufficient data: 288, Routine: 2267
- Invitations sent: **1790** (opt-outs respected: 137)
- Completed visits/screenings after outreach: **943** (53% of invited)
- Genetic-counselling referrals proposed (clinician decides): **90**
- Cascade-screening proposals: **53 families, 138 relatives** (consented, clinician-led)
- Downstream value: **59** truly high-risk people reached and seen among those flagged (simulated ground truth)

## Equity and safeguards by region (cells under 10 are suppressed)
| Segment | n | Flagged | Invited | Attended | False-positive | False-negative |
|---|---|---|---|---|---|---|
| Abu Dhabi City | 1648 | 18% | 60% | 51% | 69% | 51% |
| Al Ain | 906 | 10% | 59% | 53% | 68% | 62% |
| Al Dhafra | 446 | 22% | 59% | 58% | 72% | 49% |

## Equity and safeguards by preferred language
| Segment | n | Flagged | Invited | Attended | False-positive | False-negative |
|---|---|---|---|---|---|---|
| Arabic | 1116 | 18% | 60% | 59% | 64% | 47% |
| English | 1158 | 15% | 59% | 54% | 73% | 58% |
| Hindi | 256 | 14% | 61% | 47% | 73% | 57% |
| Urdu | 470 | 18% | 59% | 37% | 74% | 59% |

## Model calibration (predicted vs observed high-risk rate)
| Predicted band | n | Mean predicted | Observed |
|---|---|---|---|
| 0.00-0.10 | 2173 | 0.05 | 0.06 |
| 0.10-0.25 | 94 | 0.10 | 0.05 |
| 0.25-0.50 | 355 | 0.35 | 0.25 |
| 0.50-1.00 | 90 | 0.57 | 0.50 |

## Automatic equity alerts
- Uptake for **Urdu** speakers is 37% vs 57% for Arabic/English. These invitations currently go out in English because no reviewed Urdu template exists. Action: commission reviewed Urdu templates and add community health worker outreach.
- Uptake for **Hindi** speakers is 47% vs 57% for Arabic/English. These invitations currently go out in English because no reviewed Hindi template exists. Action: commission reviewed Hindi templates and add community health worker outreach.

## Example explanation cards
```
CASE P0040 (child, Abu Dhabi City, Arabic)
  Consent scope : {'visit_history': True, 'clinical': True, 'genomic': False, 'outreach': True}
  Risk tier     : High  (prob=0.832, confidence=Medium)
    + Family history: early cardiac (+2.3 log-odds)
    + Family history: hereditary cancer (+2.3 log-odds)
  Missing data  : genomic data (no consent or no test)
  Pathway       : Priority Well-Child Visit + genetic counselling referral
  Status/action : Overdue (22 months) -> Invite now
  Counselling   : offer referral (clinician decides)
  Outreach      : Community health worker in Arabic to parent/caregiver
  Message       : مرحباً، معكم نكست جن هيلث في أبوظبي. قد يحين موعد زيارة الطفل السليم لطفلكم. هذه دعوة وليست تشخيصاً. يمكنكم الحجز أو التأجيل أو الرفض في أي وقت. أرسلوا: حجز، لاحقاً، أو إيقاف.
  Note          : Population prioritisation, not a diagnosis. A clinician confirms the individual decision.
```
```
CASE P0015 (adult, Al Dhafra, Urdu)
  Consent scope : {'visit_history': True, 'clinical': True, 'genomic': False, 'outreach': True}
  Risk tier     : Elevated  (prob=0.332, confidence=Medium)
    + Family history: hemoglobinopathy (+2.3 log-odds)
  Missing data  : genomic data (no consent or no test)
  Pathway       : Earlier adult comprehensive screening
  Status/action : Overdue (31 months) -> Invite now
  Counselling   : not indicated by prioritisation
  Outreach      : Community health worker in English to adult
  Message       : Hello, this is NextGen Health, Abu Dhabi. You may be due for a comprehensive health screening. This is an invitation, not a diagnosis. You can book, postpone or decline at any time. Reply BOOK, LATER or STOP.
  ! Preferred language needs a reviewed translation (currently English)
  Note          : Population prioritisation, not a diagnosis. A clinician confirms the individual decision.
```
```
CASE P0001 (adult, Abu Dhabi City, English)
  Consent scope : {'visit_history': True, 'clinical': False, 'genomic': False, 'outreach': False}
  Risk tier     : Insufficient data  (prob=None, confidence=Low)
  Missing data  : clinical data (no consent), genomic data (no consent or no test)
  Pathway       : Adult comprehensive screening
  Status/action : Not yet eligible -> Re-check at age 30
  Counselling   : not indicated by prioritisation
  Outreach      : none (No outreach consent)
  Note          : Population prioritisation, not a diagnosis. A clinician confirms the individual decision.
```