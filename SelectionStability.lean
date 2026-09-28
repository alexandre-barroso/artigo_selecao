import Mathlib.Tactic

/- Supporting obligations for the unselected calibrated-selection risk note.
This module does not formalize sampling laws, adaptive relative entropy,
minimax risk, the Python implementation, or publication novelty. -/
namespace SelectionStability
noncomputable section

def clip (m M x : ℝ) : ℝ := min M (max m x)

theorem clip_mem (m M x : ℝ) (hmM : m ≤ M) :
    m ≤ clip m M x ∧ clip m M x ≤ M := by
  exact ⟨le_min hmM (le_max_left _ _), min_le_left _ _⟩

theorem clip_distance (m M x b : ℝ) (hb : m ≤ b) (hbM : b ≤ M) :
    |clip m M x - b| ≤ |x - b| := by
  have hmM : m ≤ M := le_trans hb hbM
  by_cases hxm : x < m
  · rw [clip, max_eq_left (le_of_lt hxm), min_eq_right hmM,
        abs_of_nonpos (by linarith : m-b ≤ 0),
        abs_of_nonpos (by linarith : x-b ≤ 0)]
    linarith
  · have hmx : m ≤ x := le_of_not_gt hxm
    by_cases hMx : M < x
    · rw [clip, max_eq_right hmx, min_eq_left (le_of_lt hMx),
          abs_of_nonneg (by linarith : 0 ≤ M-b),
          abs_of_nonneg (by linarith : 0 ≤ x-b)]
      linarith
    · rw [clip, max_eq_right hmx, min_eq_right (le_of_not_gt hMx)]

def clippedRatio (m M U D : ℝ) : ℝ :=
  if D = 0 then (m+M)/2 else clip m M (U/D)

theorem clippedRatio_mem (m M U D : ℝ) (hmM : m ≤ M) :
    m ≤ clippedRatio m M U D ∧ clippedRatio m M U D ≤ M := by
  unfold clippedRatio
  split_ifs
  · constructor <;> linarith
  · exact clip_mem m M (U/D) hmM

theorem interval_distance_bound (m M x b : ℝ)
    (hx : m ≤ x) (hxM : x ≤ M) (hb : m ≤ b) (hbM : b ≤ M) :
    |x-b| ≤ M-m := by
  apply abs_le.mpr
  constructor <;> linarith

theorem projected_residual (m M U D b : ℝ) (hb : m ≤ b) (hbM : b ≤ M) :
    |D| * |clippedRatio m M U D - b| ≤ |U-b*D| := by
  by_cases hD : D=0
  · simp [hD]
  · have h := mul_le_mul_of_nonneg_left (clip_distance m M (U/D) b hb hbM) (abs_nonneg D)
    have hid : D*(U/D-b)=U-b*D := by field_simp
    simpa [clippedRatio, hD, ← abs_mul, hid] using h

theorem projected_product_bound (m M U D Uh Dh b : ℝ)
    (hb : m ≤ b) (hbM : b ≤ M) (hU : U=b*D) :
    |D| * |clippedRatio m M Uh Dh - b| ≤
      |Uh-U| + (|b| + M-m)*|Dh-D| := by
  let bh := clippedRatio m M Uh Dh
  have hmem := clippedRatio_mem m M Uh Dh (le_trans hb hbM)
  have he : |bh-b| ≤ M-m := interval_distance_bound m M bh b hmem.1 hmem.2 hb hbM
  have hres : |Dh| * |bh-b| ≤ |Uh-b*Dh| := projected_residual m M Uh Dh b hb hbM
  have hD : |D| ≤ |Dh| + |Dh-D| := by
    calc
      |D| = |Dh+(D-Dh)| := by congr 1; ring
      _ ≤ |Dh| + |D-Dh| := abs_add_le _ _
      _ = |Dh| + |Dh-D| := by rw [abs_sub_comm D Dh]
  have hres2 : |Uh-b*Dh| ≤ |Uh-U| + |b| * |Dh-D| := by
    calc
      |Uh-b*Dh| = |(Uh-U)+(-b)*(Dh-D)| := by rw [hU]; congr 1; ring
      _ ≤ |Uh-U| + |(-b)*(Dh-D)| := abs_add_le _ _
      _ = |Uh-U| + |b| * |Dh-D| := by rw [abs_mul, abs_neg]
  have hleft := mul_le_mul_of_nonneg_right hD (abs_nonneg (bh-b))
  have hterm := mul_le_mul_of_nonneg_left he (abs_nonneg (Dh-D))
  change |D| * |bh-b| ≤ _
  nlinarith

def ratio (C a b k : ℝ) : ℝ := C*(1+k*a)/(1+k*b)
def firstDiff (C a b k1 k2 : ℝ) : ℝ :=
  (ratio C a b k2-ratio C a b k1)/(k2-k1)
def denominator (C a b k1 k2 k3 : ℝ) : ℝ :=
  k3*firstDiff C a b k2 k3-k1*firstDiff C a b k1 k2
def numerator (C a b k1 k2 k3 : ℝ) : ℝ :=
  firstDiff C a b k1 k2-firstDiff C a b k2 k3

theorem firstDiff_identity (C a b k1 k2 : ℝ)
    (hb : 0<b) (hk1 : 0<k1) (hk2 : k1<k2) :
    firstDiff C a b k1 k2 = C*(a-b)/((1+k1*b)*(1+k2*b)) := by
  have h2 : 0<k2 := lt_trans hk1 hk2
  have h1b : 1+k1*b ≠ 0 := ne_of_gt (by positivity)
  have h2b : 1+k2*b ≠ 0 := ne_of_gt (by positivity)
  have hd : k2-k1 ≠ 0 := ne_of_gt (sub_pos.mpr hk2)
  unfold firstDiff ratio
  field_simp
  ring

theorem denominator_identity (C a b k1 k2 k3 : ℝ)
    (hb : 0<b) (hk1 : 0<k1) (h12 : k1<k2) (h23 : k2<k3) :
    denominator C a b k1 k2 k3 =
      C*(a-b)*(k3-k1)/((1+k1*b)*(1+k2*b)*(1+k3*b)) := by
  have hk2 : 0<k2 := lt_trans hk1 h12
  have hk3 : 0<k3 := lt_trans hk2 h23
  have h1b : 1+k1*b ≠ 0 := ne_of_gt (by positivity)
  have h2b : 1+k2*b ≠ 0 := ne_of_gt (by positivity)
  have h3b : 1+k3*b ≠ 0 := ne_of_gt (by positivity)
  unfold denominator
  rw [firstDiff_identity C a b k2 k3 hb hk2 h23,
      firstDiff_identity C a b k1 k2 hb hk1 h12]
  field_simp
  ring

theorem numerator_identity (C a b k1 k2 k3 : ℝ)
    (hb : 0<b) (hk1 : 0<k1) (h12 : k1<k2) (h23 : k2<k3) :
    numerator C a b k1 k2 k3 = b*denominator C a b k1 k2 k3 := by
  have hk2 : 0<k2 := lt_trans hk1 h12
  have hk3 : 0<k3 := lt_trans hk2 h23
  have h1b : 1+k1*b ≠ 0 := ne_of_gt (by positivity)
  have h2b : 1+k2*b ≠ 0 := ne_of_gt (by positivity)
  have h3b : 1+k3*b ≠ 0 := ne_of_gt (by positivity)
  unfold numerator
  rw [denominator_identity C a b k1 k2 k3 hb hk1 h12 h23,
      firstDiff_identity C a b k2 k3 hb hk2 h23,
      firstDiff_identity C a b k1 k2 hb hk1 h12]
  field_simp
  ring

theorem slope_denominator_identity (C a b k1 k2 k3 : ℝ)
    (hb : 0<b) (hk1 : 0<k1) (h12 : k1<k2) (h23 : k2<k3) :
    (k3-k1)*firstDiff C a b k1 k2 =
      (1+k3*b)*denominator C a b k1 k2 k3 := by
  have hk2 : 0<k2 := lt_trans hk1 h12
  have hk3 : 0<k3 := lt_trans hk2 h23
  have h1b : 1+k1*b ≠ 0 := ne_of_gt (by positivity)
  have h2b : 1+k2*b ≠ 0 := ne_of_gt (by positivity)
  have h3b : 1+k3*b ≠ 0 := ne_of_gt (by positivity)
  rw [denominator_identity C a b k1 k2 k3 hb hk1 h12 h23,
      firstDiff_identity C a b k1 k2 hb hk1 h12]
  field_simp

theorem proposal_intercept_identity (C a b k1 k2 : ℝ)
    (hb : 0<b) (hk1 : 0<k1) (h12 : k1<k2) :
    (k2*ratio C a b k1-k1*ratio C a b k2)/(k2-k1)
      -k1*k2*firstDiff C a b k1 k2*b = C := by
  have hk2 : 0<k2 := lt_trans hk1 h12
  have h1b : 1+k1*b ≠ 0 := ne_of_gt (by positivity)
  have h2b : 1+k2*b ≠ 0 := ne_of_gt (by positivity)
  have hd : k2-k1 ≠ 0 := ne_of_gt (sub_pos.mpr h12)
  unfold firstDiff ratio
  field_simp
  ring

theorem scalar_cancellation_bound (c d dmax A D v r : ℝ)
    (hc : 0<c) (hd : 0≤d) (hdmax : d≤dmax)
    (hrel : c*A=d*D) (herr : |D| * |v| ≤ r) :
    |A| * |v| ≤ dmax/c*r := by
  have habs : c*|A| = d*|D| := by
    have h := congrArg abs hrel
    simpa [abs_mul, abs_of_pos hc, abs_of_nonneg hd] using h
  have hA : |A| = (d/c)*|D| := by
    apply mul_left_cancel₀ (ne_of_gt hc)
    calc
      c*|A| = d*|D| := habs
      _ = c*((d/c)*|D|) := by field_simp
  calc
    |A| * |v| = (d/c)*(|D| * |v|) := by rw [hA]; ring
    _ ≤ (dmax/c)*r := mul_le_mul
      (div_le_div_of_nonneg_right hdmax (le_of_lt hc)) herr
      (mul_nonneg (abs_nonneg D) (abs_nonneg v))
      (div_nonneg (le_trans hd hdmax) (le_of_lt hc))

theorem affine_intercept_error (E Eh A Ah b bh L M LE LA K e : ℝ)
    (hL : 0≤L) (hM : 0≤M) (he : 0≤e)
    (hE : |Eh-E| ≤ LE*e) (hA : |Ah-A| ≤ LA*e)
    (hb : |bh| ≤ M) (hprod : |A| * |bh-b| ≤ K*e) :
    |(Eh-L*Ah*bh)-(E-L*A*b)| ≤ (LE+L*(M*LA+K))*e := by
  have hsub : |(Ah-A)*bh+A*(bh-b)| ≤ |Ah-A| * |bh| + |A| * |bh-b| := by
    simpa [abs_mul] using abs_add_le ((Ah-A)*bh) (A*(bh-b))
  have hfirst : |Ah-A| * |bh| ≤ LA*e*M :=
    mul_le_mul hA hb (abs_nonneg bh) (by linarith [abs_nonneg (Ah-A)])
  have hwhole : |(Eh-L*Ah*bh)-(E-L*A*b)| ≤
      |Eh-E| + L*|(Ah-A)*bh+A*(bh-b)| := by
    calc
      _ = |(Eh-E)+(-L)*((Ah-A)*bh+A*(bh-b))| := by congr 1; ring
      _ ≤ |Eh-E| + |(-L)*((Ah-A)*bh+A*(bh-b))| := abs_add_le _ _
      _ = _ := by rw [abs_mul, abs_neg, abs_of_nonneg hL]
  have hscaled := mul_le_mul_of_nonneg_left hsub hL
  have hfscaled := mul_le_mul_of_nonneg_left hfirst hL
  have hpscaled := mul_le_mul_of_nonneg_left hprod hL
  nlinarith

theorem absolute_perturbation (x y e : ℝ) (h : |x-y| ≤ e) :
    |x| ≤ |y| + e := by
  calc
    |x| = |y+(x-y)| := by congr 1; ring
    _ ≤ |y| + |x-y| := abs_add_le _ _
    _ ≤ |y| + e := by linarith

theorem selected_pair_bound {n : ℕ} (d dh : Fin n → ℝ) (chosen i : Fin n)
    (v W K e : ℝ) (he : 0≤e)
    (herror : ∀ j, |dh j-d j| ≤ e)
    (hselected : ∀ j, |dh j| ≤ |dh chosen|)
    (hproduct : |d chosen| * |v| ≤ K*e) (hbounded : |v| ≤ W) :
    |d i| * |v| ≤ (K+2*W)*e := by
  have hi : |d i| ≤ |dh i| + e := by
    apply absolute_perturbation
    simpa [abs_sub_comm] using herror i
  have hc : |dh chosen| ≤ |d chosen| + e :=
    absolute_perturbation _ _ _ (herror chosen)
  have hsel := hselected i
  have hm : |d i| ≤ |d chosen| + 2*e := by linarith
  have hv := mul_le_mul_of_nonneg_right hm (abs_nonneg v)
  have hw := mul_le_mul_of_nonneg_left hbounded (by linarith : 0≤2*e)
  nlinarith

theorem calibration_fraction_bound (k m t u : ℝ)
    (hk : 0<k) (hm : 0<m) (ht : m≤t) (hu : m≤u) :
    k^2/((1+k*t)*(1+k*u)) ≤ 1/m^2 := by
  have htpos : 0<t := lt_of_lt_of_le hm ht
  have hupos : 0<u := lt_of_lt_of_le hm hu
  have hp : 0<(1+k*t)*(1+k*u) := by positivity
  apply (div_le_div_iff₀ hp (pow_pos hm 2)).mpr
  have ht' : k*m≤1+k*t := by nlinarith
  have hu' : k*m≤1+k*u := by nlinarith
  calc
    k^2*m^2 = (k*m)*(k*m) := by ring
    _ ≤ (1+k*t)*(1+k*u) := mul_le_mul ht' hu' (by positivity) (by positivity)
    _ = 1*((1+k*t)*(1+k*u)) := by ring

#print axioms clip_mem
#print axioms clip_distance
#print axioms clippedRatio_mem
#print axioms interval_distance_bound
#print axioms projected_residual
#print axioms projected_product_bound
#print axioms firstDiff_identity
#print axioms denominator_identity
#print axioms numerator_identity
#print axioms slope_denominator_identity
#print axioms proposal_intercept_identity
#print axioms scalar_cancellation_bound
#print axioms affine_intercept_error
#print axioms absolute_perturbation
#print axioms selected_pair_bound
#print axioms calibration_fraction_bound

end
end SelectionStability
