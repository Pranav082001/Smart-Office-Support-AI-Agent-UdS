// Only allow a team email to be entered once its custom category has a name.
document.querySelectorAll(".custom-category-row").forEach((row) => {
  const catInput = row.querySelector(".custom-cat-input");
  const roleInput = row.querySelector(".custom-role-input");

  catInput.addEventListener("input", () => {
    const hasCategory = catInput.value.trim() !== "";
    roleInput.readOnly = !hasCategory;
    if (!hasCategory) {
      roleInput.value = "";
    }
  });
});
