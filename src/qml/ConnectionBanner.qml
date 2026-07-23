import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Material 2.12
import QtQuick.Layouts 1.12

Rectangle {
    id: root
    color: theme.background

    property string message: ""

    signal checkConnection()

    Theme { id: theme }

    ColumnLayout {
        anchors.centerIn: parent
        spacing: 16
        width: Math.min(parent.width - 80, 420)

        Label {
            text: "⚠"
            font.pixelSize: 48
            color: theme.danger
            Layout.alignment: Qt.AlignHCenter
        }
        Label {
            text: "No EVK1 detected."
            color: theme.text
            font.pixelSize: 18
            font.bold: true
            Layout.alignment: Qt.AlignHCenter
        }
        Label {
            text: root.message.length > 0 ? root.message : "Plug in the camera, then press Check Connection."
            color: theme.mutedText
            wrapMode: Text.WordWrap
            horizontalAlignment: Text.AlignHCenter
            Layout.fillWidth: true
            Layout.alignment: Qt.AlignHCenter
        }
        Button {
            text: "Check Connection"
            highlighted: true
            Layout.alignment: Qt.AlignHCenter
            onClicked: root.checkConnection()
        }
    }
}
