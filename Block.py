from dataclasses import dataclass, field

# --- Block tracking struct ---
@dataclass
class Block:
    id: int
    last5box: list = field(default_factory=list)  # up to 5 [x1,y1,x2,y2] normalized bboxes

    def update(self, bbox_norm):
        """Append a new normalized [x1,y1,x2,y2] bbox, keeping only the last 5."""
        self.last5box.append(bbox_norm)
        if len(self.last5box) > 5:
            self.last5box.pop(0)