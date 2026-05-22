# Retrieval Access Policy

## Intended Rules

| Department | Allowed Document Classes | Restricted Document Classes |
|------------|--------------------------|-----------------------------|
| HR | HR internal, manager handbook, approved policy documents | Compensation planning, investigations, executive HR notes |
| Finance | Finance internal, expense policy, approval matrix | Board prep, unreleased forecasts, payroll detail |
| Engineering | Engineering internal, incident guides, runbooks | Security incident details, unreleased architecture plans |
| Operations | Operations internal, vendor playbooks | Contract negotiations, executive planning |

## Policy Requirements

1. Users may retrieve documents only from their department unless an explicit group grants access.
2. Confidential documents require both department match and allowed group match.
3. Restricted documents should never be summarized unless the user belongs to a named privileged group.
4. Citations should not reveal titles or IDs of inaccessible documents.
5. Export destinations must be owned by the same department as the requesting user.
